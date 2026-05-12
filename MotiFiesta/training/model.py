import random
from collections import defaultdict
from collections import Counter
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from torch.nn.functional import normalize
from sklearn.neighbors import KernelDensity as KDE
from sklearn.neighbors import KDTree
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from MotiFiesta.training.edge_pool import EdgePooling
from MotiFiesta.utils.subgraph_similarity import build_wwl_K

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.graph_utils import (
    spotlight_at,
    edge_spotlight,
    spotlight_key,
    get_edge_subgraphs_tensor,
)


class MotiFiestaModel(torch.nn.Module):
    """GCN model that iteratively applies edge contraction and computes node embeddings.
    """
    def __init__(self,
                 n_features=16,
                 dim=32,
                 steps=5,
                 conv=None,
                 hard_embed=False,
                 pool_dummy=None,
                 merge_method='sum',
                 global_pool=global_add_pool,
                 edge_score_method='sigmoid',
                 parallel_matching=False,
                 ):
        super(MotiFiestaModel, self).__init__()

        self.steps = steps
        self.n_features = n_features
        self.hidden_dim = dim
        self.steps = steps
        self.hard_embed = hard_embed
        self.edge_score_method = edge_score_method
        self.merge_method = merge_method
        self.hard_embed = hard_embed
        self.parallel_matching = parallel_matching

        self.layers = self.build_layers()

    def build_layers(self):
        layers = []
        layers.append(EdgePooling(self.n_features,
                                  self.hidden_dim,
                                  edge_score_method=self.edge_score_method,
                                  merge_method=self.merge_method,
                                  parallel_matching=self.parallel_matching,
                                  ))
        for s in range(self.steps):
            layers.append(EdgePooling(self.hidden_dim,
                                      self.hidden_dim,
                                      edge_score_method=self.edge_score_method,
                                      merge_method=self.merge_method,
                                      parallel_matching=self.parallel_matching,
                                      )
                          )

        return torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, batch, n_id=None, dummy=False, x_null=None, e_null=None):
        """One forward pass applies the model over all steps."""
        if n_id is None:
            n_id = torch.arange(len(x), device=x.device)

        n_batch_nodes = x.size(0)

        spotlight_assignment = [
            torch.arange(n_batch_nodes, device=x.device, dtype=torch.long)
        ]
        cluster_chain = []

        xx, ee, pp = [], [], []
        batches = []
        internals = []

        for t, layer in enumerate(self.layers):

            if len(edge_index[0]) < 1:
                break
            out = layer(x, edge_index, batch, hard_embed=self.hard_embed, dummy=dummy)

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

        merge_info = {
            'spotlight_assignment': spotlight_assignment,
            'cluster_chain': cluster_chain,
            'n_id': n_id,
        }

        return xx, pp, ee, batches, merge_info, internals

    def rec_loss(self, xx, ee, merge_info, source_graph, internals,
                 num_nodes=20, draw=False):
        """reconstruction loss at all coarsening levels using the wwl kernel."""
        source_ig = source_graph.ig_graph
        source_x = source_graph.cached_data.x

        spot_assign = merge_info['spotlight_assignment']
        n_id = merge_info['n_id']

        loss = 0
        for level in range(len(xx)):
            x = internals[level]['x_merged']

            subgraphs, node_features = get_edge_subgraphs_tensor(
                ee[level], spot_assign, n_id, level, source_ig, source_x,
            )

            K_predict = matrix_cosine(x[:num_nodes], x[:num_nodes])
            K_predict = K_predict.to(get_device())

            formatted_features = [
                (f.cpu().numpy() if torch.is_tensor(f) else f).astype(np.float64)
                for f in node_features[:num_nodes]
            ]

            K_true = build_wwl_K(subgraphs[:num_nodes], formatted_features)
            K_true = K_true.to(get_device())

            if draw:
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        g1, g2 = subgraphs[i], subgraphs[j]
                        fig, ax = plt.subplots(1, 2)
                        nx.draw(g1, ax=ax[0])
                        nx.draw(g2, ax=ax[1])
                        fig.suptitle(f"true: {K_true[i][j]}, pred: {K_predict[i][j]}")
                        plt.show()

            l = torch.nn.MSELoss()(K_predict, K_true)
            loss += l

        return loss / self.steps

    def rec_loss_wl(self, xx, ee, merge_info, source_graph, internals,
                    num_nodes=20, edge_sample_rate=1.0, wl_iter=3):
        """edge-level reconstruction loss using the wl subtree kernel."""
        from MotiFiesta.training.wl_kernel import (
            wl_subtree_similarity_batch,
            initial_labels_from_onehot,
        )
        from torch_geometric.utils import subgraph as pyg_subgraph

        device = get_device()

        source_edge_index = source_graph.cached_data.edge_index.to(device)
        source_x = source_graph.cached_data.x.to(device)
        source_labels = initial_labels_from_onehot(source_x)
        n_source = source_x.size(0)

        spot_assign = merge_info['spotlight_assignment']
        n_id = merge_info['n_id']

        loss = 0
        for level in range(len(xx)):
            x = internals[level]['x_merged']
            n_take = min(num_nodes, x.size(0))
            if n_take < 2:
                continue

            spot_t = spot_assign[level]
            edge_idx_at_level = ee[level]

            edge_indices = []
            node_counts = []
            init_labels = []
            valid_local = []

            for k in range(n_take):
                u_idx = edge_idx_at_level[0, k].item()
                v_idx = edge_idx_at_level[1, k].item()

                mask = (spot_t == u_idx) | (spot_t == v_idx)
                local_members = mask.nonzero(as_tuple=False).squeeze(-1)
                if local_members.numel() == 0:
                    continue

                global_node_idx = n_id[local_members].sort().values

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
            K_true[valid_idx.unsqueeze(1), valid_idx.unsqueeze(0)] = K_valid

            K_predict = matrix_cosine(x[:n_take], x[:n_take]).to(device)

            if edge_sample_rate < 1.0:
                mask = torch.rand(n_take, n_take, device=device) < edge_sample_rate
                mask = mask | mask.t()
                mask.fill_diagonal_(True)
                K_predict = K_predict * mask.float()
                K_true = K_true * mask.float()

            l = torch.nn.MSELoss()(K_predict, K_true)
            loss += l

        return loss / self.steps

    @staticmethod
    def kde(X, X_ref, h=1):
        kde = KDE(kernel='gaussian', bandwidth=h).fit(X_ref.detach().numpy())
        f = kde.score_samples(X.detach().numpy())
        f = torch.tensor(f, dtype=torch.float32)
        return torch.exp(f)

    @staticmethod
    def distance_density(X, X_ref, k=20, max_ref=2000):
        """distance to k-th nearest neighbor in X_ref for each row of X.

        torch-native via pairwise distances + topk. no cpu transfer, no
        numpy, no sklearn. matches the original sklearn-based version's
        no-gradient semantics (densities are targets, not differentiable).
        """
        with torch.no_grad():
            if X_ref.size(0) > max_ref:
                sample = torch.randperm(X_ref.size(0), device=X_ref.device)[:max_ref]
                X_ref = X_ref[sample]
            dists = torch.cdist(X, X_ref)
            R, _ = dists.topk(k, dim=1, largest=False)
            return R[:, k - 1].detach()

    @staticmethod
    def knn_density(X, X_ref, volume=False, epsilon=1e-5, k=50):
        d = X.shape[1]
        N = X_ref.shape[0]
        knn = KDTree(X_ref.cpu().detach().numpy())

        R,_ = knn.query(X.cpu().detach().numpy(), k=k)
        R = R[:,k-1]
        if volume:
            V = ((np.pi**(d/2)) / gamma(d/2 +1)) * (R**d)
            f_hat = (k / N) * (1 / V)
        else:
            f_hat = R

        f_hat = torch.tensor(f_hat, dtype=torch.float32, requires_grad=False)
        f_hat = f_hat.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        return f_hat

    @staticmethod
    def min_density(X, X_ref, epsilon=1e-5, k=10):
        d = torch.cdist(X, X_ref)
        return 1/d.min(dim=1)[0]

    def freq_loss(self, internals_pos, internals_neg, pp,
                  estimator='knn', beta=1, lam=1, steps=3,
                  volume=False, k=30):
        """penalize embeddings close to randos or sparse."""
        tot_loss = 0
        for t in range(len(pp)):
            x_pos = internals_pos[t]['x_merged']
            x_neg = internals_neg[t]['x_merged']
            s = pp[t]

            k_eff = min(k, x_pos.size(0), x_neg.size(0))
            if k_eff < 2:
                continue

            if estimator == 'kde':
                density_pos = self.kde(x_pos, x_pos)
                density_neg = self.kde(x_pos, x_neg)
            if estimator == 'knn':
                density_pos = self.distance_density(x_pos, x_pos, k=k_eff)
                density_neg = self.distance_density(x_pos, x_neg, k=k_eff)

                scale = torch.cat([density_pos, density_neg]).max() + 1e-8
                density_pos = density_pos / scale
                density_neg = density_neg / scale
            if estimator == 'min':
                density_pos = self.min_density(x_pos, x_pos)
                density_neg = self.min_density(x_pos, x_neg)

            f_pos = density_pos.view(-1, 1).squeeze()
            f_neg = density_neg.view(-1, 1).squeeze()

            l = (-1 * s * torch.exp(-1 * beta * (f_pos - f_neg))).mean()
            reg_term = lam * s.pow(2.0).mean()
            l += reg_term

            tot_loss += l

        tot_loss /= steps
        return tot_loss

    def sil_loss(self, internals_pos, merge_info, tracker, momentum=0.95,
                 max_per_level=100):
        """sampling invariance loss with per-level sampling cap.

        original semantics: track an ema target embedding for every
        (spotlight, level) pair, penalize current-batch deviations.
        large graphs produce thousands of supernodes per level, and the
        python-bound iteration dominates wall time. capping the number of
        supernodes processed per level per batch yields ~10x speedup; the
        ema update has enough cross-batch coverage to compensate.

        :param max_per_level: maximum number of supernodes to update per
            level per batch. set high (e.g. 10000) to recover the
            non-sampled behavior. default 100 gives ~10x speedup.
        """
        device = get_device()
        spot_assign = merge_info['spotlight_assignment']
        n_id = merge_info['n_id']

        tot_loss = torch.zeros(1, device=device)
        n_terms = 0

        for level in range(len(internals_pos)):
            x_level = internals_pos[level]['x_merged']
            if x_level.size(0) == 0:
                continue

            h_cur = F.normalize(x_level, dim=-1)
            spot_t = spot_assign[level]
            n_x = x_level.size(0)

            # group local member indices by their supernode id at this level.
            # sort once, find contiguous spans via unique_consecutive.
            sorted_idx = spot_t.argsort()
            spot_sorted = spot_t[sorted_idx]
            unique_vals, counts = torch.unique_consecutive(
                spot_sorted, return_counts=True)

            # filter to supernode ids that lie within x_level's row range
            valid_mask = unique_vals < n_x
            unique_vals_valid = unique_vals[valid_mask]

            # compute the starting offset for each unique value's span
            cumcounts = counts.cumsum(0)
            starts = torch.cat([
                torch.zeros(1, dtype=cumcounts.dtype, device=device),
                cumcounts[:-1]
            ])
            starts_valid = starts[valid_mask]
            counts_valid = counts[valid_mask]

            # subsample if too many supernodes to process this level
            n_valid = unique_vals_valid.size(0)
            if n_valid > max_per_level:
                sample_perm = torch.randperm(n_valid, device=device)[:max_per_level]
                unique_vals_valid = unique_vals_valid[sample_perm]
                starts_valid = starts_valid[sample_perm]
                counts_valid = counts_valid[sample_perm]

            # tight loop over the (possibly sampled) valid supernodes only
            unique_vals_list = unique_vals_valid.tolist()
            starts_list = starts_valid.tolist()
            counts_list = counts_valid.tolist()

            for k_val, start, count in zip(unique_vals_list, starts_list,
                                           counts_list):
                local_members = sorted_idx[start:start + count]
                global_ids = n_id[local_members].sort().values
                key = spotlight_key(global_ids)

                target = tracker.get(key, level)
                if target is None:
                    tracker.set(key, level, h_cur[k_val].detach())
                    continue

                target = target.to(device)
                tot_loss = tot_loss + (1.0 - (h_cur[k_val] * target).sum())
                n_terms += 1

                new_target = momentum * target + (1.0 - momentum) * h_cur[k_val].detach()
                new_target = F.normalize(new_target, dim=-1)
                tracker.set(key, level, new_target)

        if n_terms == 0:
            return torch.zeros(1, device=device).squeeze()
        return (tot_loss / n_terms).squeeze()


class SamplingInvarianceTracker:
    """momentum-updated target embeddings keyed by (spotlight, level)."""
    def __init__(self):
        self.store = {}

    def get(self, key, level):
        return self.store.get((key, level))

    def set(self, key, level, value):
        self.store[(key, level)] = value.detach().cpu()

    def __len__(self):
        return len(self.store)


def matrix_cosine(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class HardEmbedder(torch.nn.Module):
    def __init__(self, out_dim):
        super(HardEmbedder, self).__init__()
        self.out_dim = out_dim

    def forward(self, t, spotlights, edge_index_initial, nodes_initial):
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes_initial)))
        G.add_edges_from(zip(*edge_index_initial.cpu().detach().numpy()))

        edge_index_initial.to(get_device())

        def spotlight_graph(node):
            return G.subgraph(spotlights[t][node]).copy()

        embeddings = []
        for pool_node in range(len(spotlights[t])):
            subg = spotlight_graph(pool_node)
            if t == 0:
                degs = Counter((G.degree(n) for n in subg.nodes()))
            else:
                degs = Counter((subg.degree(n) for n in subg.nodes()))

            deg_hist = [degs[ind] for ind in range(self.out_dim)]
            embeddings.append(torch.Tensor(deg_hist))

        embeddings = torch.stack(embeddings)
        return embeddings


def plot_K(K_true, K_pred):
    fig, ax = plt.subplots(1, 2)
    sns.heatmap(K_true.detach().numpy(), vmin=0, vmax=1, ax=ax[0])
    sns.heatmap(K_pred.detach().numpy(), vmin=0, vmax=1, ax=ax[1])
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
