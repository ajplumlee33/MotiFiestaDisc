from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from MotiFiesta.training.edge_pool import EdgePooling
from MotiFiesta.utils.learning_utils import get_device


def _gin_mlp(in_dim, out_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim),
    )


def matrix_cosine(a, b, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    b_n = b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class DiscModel(torch.nn.Module):
    """EdgePool motif discovery with GIN convolution before each pooling level.

    GIN enriches each node's embedding with neighborhood context before each
    pooling decision, giving score_net a level-appropriate structural signal at
    every coarsening depth. GIN's WL-equivalent expressiveness also strengthens
    rec_loss_wl — the architecture is predisposed toward the WL similarity
    targets rather than fighting them through a linear bottleneck.

    interface is identical to MotiFiestaModel: forward() returns the same tuple,
    and freq_loss / rec_loss_wl have the same signatures.
    """

    def __init__(
        self,
        n_features=61,
        dim=8,
        steps=5,
        hard_embed=False,
        matching_mode='luby',
        scoring_mode='mlp',
        n_heads=4,
        **kwargs,  # absorb motifiesta-only hparams (pool_dummy, edge_score_method, etc.)
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = dim
        self.steps = steps
        self.hard_embed = hard_embed
        self.matching_mode = matching_mode
        self.scoring_mode = scoring_mode
        self.n_heads = n_heads

        # level 0: project in_features → dim; levels 1-steps: dim → dim
        self.gin_layers = torch.nn.ModuleList([
            GINConv(_gin_mlp(n_features, dim)),
            *[GINConv(_gin_mlp(dim, dim)) for _ in range(steps)],
        ])

        # level 0 pool receives n_features (pre-gin); levels 1+ receive dim (x_new from previous level)
        self.pool_layers = torch.nn.ModuleList([
            EdgePooling(
                n_features, dim,
                matching_mode=matching_mode,
                scoring_mode=scoring_mode,
                n_heads=n_heads,
            ),
            *[
                EdgePooling(
                    dim, dim,
                    matching_mode=matching_mode,
                    scoring_mode=scoring_mode,
                    n_heads=n_heads,
                )
                for _ in range(steps)
            ],
        ])

        # alias so load_model backward-compat and decoder work unchanged
        self.layers = self.pool_layers

    def forward(self, x, edge_index, batch, n_id=None, dummy=False, **kwargs):
        if n_id is None:
            n_id = torch.arange(len(x), device=x.device)

        n_batch_nodes = x.size(0)
        spotlight_assignment = [
            torch.arange(n_batch_nodes, device=x.device, dtype=torch.long)
        ]
        cluster_chain = []

        xx, ee, pp, batches, internals = [], [], [], [], []

        for t, (gin, pool) in enumerate(zip(self.gin_layers, self.pool_layers)):
            if len(edge_index[0]) < 1:
                break

            # dual pathway: gin enriches for scoring only; x (pre-gin) feeds transform for hash embeddings.
            # this keeps hash embeddings consistent across instances while giving score_net expressive features.
            x_gin = F.relu(gin(x, edge_index))

            out = pool(x, edge_index, batch, hard_embed=self.hard_embed, dummy=dummy, x_score=x_gin)

            ee.append(edge_index)
            xx.append(x)
            pp.append(out['internals']['edge_scores'])
            batches.append(batch)
            internals.append(out['internals'])

            cluster = out['new_graph']['unpool'].cluster
            cluster_chain.append(cluster)
            spotlight_assignment.append(cluster[spotlight_assignment[-1]])

            edge_index = out['new_graph']['e_ind_new']
            x = out['new_graph']['x_new']
            batch = out['new_graph']['batch_new']

        spotlights = []
        for sa in spotlight_assignment:
            spot_t = defaultdict(set)
            for node_idx, sup_id in enumerate(sa.tolist()):
                spot_t[sup_id].add(node_idx)
            spotlights.append(spot_t)

        tree = [defaultdict(set)]
        for cluster in cluster_chain:
            tree_t = defaultdict(set)
            for child_id, parent_id in enumerate(cluster.tolist()):
                tree_t[parent_id].add(child_id)
            tree.append(tree_t)

        merge_info = {
            'spotlight_assignment': spotlight_assignment,
            'cluster_chain': cluster_chain,
            'n_id': n_id,
            'spotlights': spotlights,
            'tree': tree,
        }

        return xx, pp, ee, batches, merge_info, internals

    def rec_loss_wl(
        self,
        xx,
        ee,
        merge_info,
        source_graph,
        internals,
        num_nodes=20,
        edge_sample_rate=1.0,
        wl_iter=3,
        max_spotlight_nodes=20,
    ):
        from MotiFiesta.training.wl_kernel import (
            wl_subtree_similarity_batch,
            initial_labels_from_onehot,
        )
        from torch_geometric.utils import subgraph as pyg_subgraph

        device = get_device()
        use_cpu_kernel = (device.type == 'mps')

        source_edge_index = source_graph.cached_data.edge_index
        source_x = source_graph.cached_data.x
        if use_cpu_kernel:
            source_edge_index = source_edge_index.cpu()
            source_x = source_x.cpu()
        else:
            source_edge_index = source_edge_index.to(device)
            source_x = source_x.to(device)
        source_labels = initial_labels_from_onehot(source_x)
        n_source = source_labels.size(0)

        spot_assign = merge_info['spotlight_assignment']
        n_id = merge_info['n_id']

        loss = 0
        for level in range(len(xx)):
            x = internals[level]['x_merged']
            n_take = min(num_nodes, x.size(0))
            if n_take < 2:
                continue

            spot_t = spot_assign[level]
            edge_idx = ee[level]
            n_id_l = n_id
            if use_cpu_kernel:
                spot_t = spot_t.cpu()
                edge_idx = edge_idx.cpu()
                n_id_l = n_id.cpu()

            edge_indices, node_counts, init_labels, valid_local = [], [], [], []

            for k in range(n_take):
                u_idx = edge_idx[0, k].item()
                v_idx = edge_idx[1, k].item()

                mask = (spot_t == u_idx) | (spot_t == v_idx)
                local_members = mask.nonzero(as_tuple=False).squeeze(-1)
                if local_members.numel() == 0:
                    continue

                global_node_idx = n_id_l[local_members].sort().values

                if global_node_idx.size(0) > max_spotlight_nodes:
                    perm = torch.randperm(global_node_idx.size(0))[:max_spotlight_nodes]
                    global_node_idx = global_node_idx[perm].sort().values

                ei_sub, _ = pyg_subgraph(
                    global_node_idx,
                    source_edge_index,
                    relabel_nodes=True,
                    num_nodes=n_source,
                )
                edge_indices.append(ei_sub)
                node_counts.append(global_node_idx.size(0))
                init_labels.append(source_labels[global_node_idx])
                valid_local.append(k)

            if len(valid_local) < 2:
                continue

            K_valid = wl_subtree_similarity_batch(
                edge_indices, node_counts, init_labels, n_iter=wl_iter,
            )

            K_true = torch.zeros(n_take, n_take, device=device)
            valid_idx = torch.tensor(valid_local, dtype=torch.long, device=device)
            K_true[valid_idx.unsqueeze(1), valid_idx.unsqueeze(0)] = K_valid.to(device)

            K_predict = matrix_cosine(x[:n_take], x[:n_take]).to(device)

            if edge_sample_rate < 1.0:
                mask = torch.rand(n_take, n_take, device=device) < edge_sample_rate
                mask = mask | mask.t()
                mask.fill_diagonal_(True)
                K_predict = K_predict * mask.float()
                K_true = K_true * mask.float()

            loss += torch.nn.MSELoss()(K_predict, K_true)

        return loss / self.steps

    @staticmethod
    def distance_density(X, X_ref, k=20, max_ref=2000):
        with torch.no_grad():
            if X_ref.size(0) > max_ref:
                sample = torch.randperm(X_ref.size(0), device=X_ref.device)[:max_ref]
                X_ref = X_ref[sample]
            dists = torch.cdist(X, X_ref)
            knn_dists, _ = dists.topk(k, dim=1, largest=False)
            return knn_dists[:, k - 1]

    def freq_loss(
        self,
        internals_pos,
        internals_neg,
        pp,
        estimator='knn',
        beta=1,
        lam=1,
        steps=3,
        volume=False,
        k=30,
        wl_weights=None,
    ):
        device = next(self.parameters()).device
        tot_loss = None
        for t in range(min(len(pp), len(internals_neg))):
            x_pos = F.normalize(internals_pos[t]['x_merged'], dim=-1)
            x_neg = F.normalize(internals_neg[t]['x_merged'], dim=-1)
            s = pp[t]
            w = wl_weights[t].to(device) if (wl_weights is not None and t < len(wl_weights)) else None

            max_e = 500
            if x_pos.size(0) > max_e:
                perm = torch.randperm(x_pos.size(0), device=x_pos.device)[:max_e]
                x_pos = x_pos[perm]
                s = s[perm]
                if w is not None:
                    w = w[perm]
            if x_neg.size(0) > max_e:
                perm = torch.randperm(x_neg.size(0), device=x_neg.device)[:max_e]
                x_neg = x_neg[perm]

            k_eff = min(k, x_pos.size(0), x_neg.size(0))
            if k_eff < 2:
                continue

            density_pos = self.distance_density(x_pos, x_pos, k=k_eff)
            density_neg = self.distance_density(x_pos, x_neg, k=k_eff)

            f_pos = density_pos.view(-1, 1).squeeze()
            f_neg = density_neg.view(-1, 1).squeeze()

            if w is not None:
                l = (-1 * w * s * torch.exp(-1 * beta * (f_pos - f_neg))).mean()
            else:
                l = (-1 * s * torch.exp(-1 * beta * (f_pos - f_neg))).mean()

            l += lam * s.pow(2.0).mean()
            tot_loss = l if tot_loss is None else tot_loss + l

        if tot_loss is None:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / steps
