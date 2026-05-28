import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_scatter import scatter_mean

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.training.scatter_pool import ScatterPool


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
    """scatter-pool motif discovery model.

    global GIN pass produces H once. ScatterPool levels segment the graph
    using Z = scatter_mean(H, assignment) for scoring.
    motif embedding at level t is Z[i] (mean of H
    over spotlight), consistent across instances by WL-equivalence.
    """

    def __init__(
        self,
        n_features=61,
        dim=8,
        steps=5,
        gin_layers=2,
        matching_mode='luby',
        hard_embed=False,
        # absorb motifiesta-only hparams
        scoring_mode=None,
        n_heads=None,
        pool_dummy=None,
        edge_score_method=None,
        merge_method=None,
        **kwargs,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = dim
        self.steps = steps
        self.gin_layers_count = gin_layers
        self.matching_mode = matching_mode

        self.global_gin = torch.nn.ModuleList([
            GINConv(_gin_mlp(n_features, dim)),
            *[GINConv(_gin_mlp(dim, dim)) for _ in range(gin_layers - 1)],
        ])

        # frozen random GIN: same architecture, fixed weights, used as structural
        # similarity target in rec_loss — replaces the WL kernel, fully GPU-native
        self.frozen_gin = torch.nn.ModuleList([
            GINConv(_gin_mlp(n_features, dim)),
            *[GINConv(_gin_mlp(dim, dim)) for _ in range(gin_layers - 1)],
        ])
        for p in self.frozen_gin.parameters():
            p.requires_grad_(False)

        self.pool_layers = torch.nn.ModuleList([
            ScatterPool(dim, matching_mode=matching_mode)
            for _ in range(steps + 1)
        ])

        # alias for load_model backward compat
        self.layers = self.pool_layers

    def forward(self, x, edge_index, batch, n_id=None, dummy=False, **kwargs):
        n_nodes = x.size(0)
        if n_id is None:
            n_id = torch.arange(n_nodes, device=x.device)

        # global GIN pass — H fixed for all levels
        h = x
        for gin in self.global_gin:
            h = F.relu(gin(h, edge_index))

        cum_assign = torch.arange(n_nodes, device=x.device, dtype=torch.long)
        edge_index_cur = edge_index
        batch_cur = batch

        cumulative_assignments = [cum_assign.clone()]  # level 0 = identity
        cluster_chain = []

        xx, pp, ee, batches, internals = [], [], [], [], []
        supernode_score_levels = []

        for pool in self.pool_layers:
            n_cur = x.size(0) if len(xx) == 0 else xx[-1].size(0)
            # recompute n_cur from cum_assign
            n_cur = cum_assign.max().item() + 1

            Z = scatter_mean(h, cum_assign, dim=0, dim_size=n_cur)

            cluster, edge_scores, edge_logits, new_ei, new_batch = pool(Z, edge_index_cur, batch_cur)

            if edge_scores.size(0) > 0:
                sup_scores = scatter_mean(edge_scores, edge_index_cur[0], dim=0, dim_size=n_cur)
            else:
                sup_scores = torch.zeros(n_cur, device=x.device)

            cum_assign = cluster[cum_assign]
            cluster_chain.append(cluster)
            cumulative_assignments.append(cum_assign)

            xx.append(Z)
            pp.append(edge_scores)
            ee.append(edge_index_cur)
            batches.append(batch_cur)
            supernode_score_levels.append(sup_scores)
            internals.append({
                'x_merged': Z,
                'edge_scores': edge_scores,
                'edge_logits': edge_logits,
                'supernode_scores': sup_scores,
                'edge_index': edge_index_cur,
            })

            edge_index_cur = new_ei
            batch_cur = new_batch

            if new_ei.size(1) == 0:
                break

        merge_info = {
            'cumulative_assignments': cumulative_assignments[:-1],  # indices match xx
            'supernode_scores': supernode_score_levels,
            'n_id': n_id,
            # kept for backward compat with total_sigma / old decoder path
            'spotlights': None,
            'tree': None,
        }

        return xx, pp, ee, batches, merge_info, internals

    def rec_loss(self, xx, ee, merge_info, batch, internals=None):
        """structural similarity loss via frozen random GIN.

        K_true  = cosine similarity of frozen GIN supernode embeddings
        K_predict = cosine similarity of trainable GIN supernode embeddings
        """
        device = get_device()
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)

        with torch.no_grad():
            h_frozen = x
            for gin in self.frozen_gin:
                h_frozen = F.relu(gin(h_frozen, edge_index))

        cum_assigns = merge_info['cumulative_assignments']

        loss = 0
        n_levels_used = 0
        for level in range(len(xx)):
            Z = xx[level]
            n_sup = Z.size(0)
            if n_sup < 2:
                continue

            cum_assign = cum_assigns[level]
            Z_frozen = scatter_mean(h_frozen, cum_assign, dim=0, dim_size=n_sup)

            K_true = matrix_cosine(Z_frozen, Z_frozen).detach()
            K_predict = matrix_cosine(Z, Z)

            loss += torch.nn.MSELoss()(K_predict, K_true)
            n_levels_used += 1

        return loss / max(n_levels_used, 1)

    @staticmethod
    def distance_density(X, X_ref, k=20, max_ref=2000):
        with torch.no_grad():
            if X_ref.size(0) > max_ref:
                sample = torch.randperm(X_ref.size(0), device=X_ref.device)[:max_ref]
                X_ref = X_ref[sample]
            dists = torch.cdist(X, X_ref)
            knn_dists, _ = dists.topk(k, dim=1, largest=False)
            return knn_dists[:, k - 1]

    def cosine_loss(self, internals_pos):
        """contrastive z-cosine supervision for score_net.

        targets are z-cosine between endpoint pairs, normalized within each
        level to zero mean/unit std before being scaled to soft [0,1] labels.
        normalization prevents the degenerate constant-prediction solution that
        arises when all raw cosines are near 1.0 after scatter_mean pooling.
        supervises pre-sigmoid logits with BCEWithLogitsLoss to avoid sigmoid
        saturation blocking gradients.
        """
        device = next(self.parameters()).device
        loss = 0
        n_levels = 0

        for t in range(len(internals_pos)):
            Z = internals_pos[t]['x_merged']
            edge_idx = internals_pos[t]['edge_index']
            edge_logits = internals_pos[t].get('edge_logits')

            if edge_idx.size(1) == 0 or edge_logits is None or edge_logits.size(0) == 0:
                continue

            z_src = F.normalize(Z[edge_idx[0]], dim=-1)
            z_dst = F.normalize(Z[edge_idx[1]], dim=-1)
            cos_target = (z_src * z_dst).sum(dim=-1).detach()

            # skip levels with no contrast — scatter_mean collapses variance at higher levels
            if cos_target.numel() < 2 or cos_target.std() < 1e-5:
                continue

            # normalize to zero mean/unit std then stretch to soft [0,1] labels
            # temperature=2: one-std difference → sigmoid output ~0.88 vs ~0.12
            target_norm = (cos_target - cos_target.mean()) / (cos_target.std() + 1e-6)
            target_soft = torch.sigmoid(target_norm * 2).detach()

            loss += F.binary_cross_entropy_with_logits(edge_logits, target_soft)
            n_levels += 1

        if n_levels == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return loss / n_levels
