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
from MotiFiesta.utils.graph_utils import *


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
                 edge_score_method='sigmoid'
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

        self.layers = self.build_layers()

    def build_layers(self):
        layers = []
        layers.append(EdgePooling(self.n_features,
                                  self.hidden_dim,
                                  edge_score_method=self.edge_score_method,
                                  merge_method=self.merge_method
                                  ))
        for s in range(self.steps):
            layers.append(EdgePooling(self.hidden_dim,
                                      self.hidden_dim,
                                      edge_score_method=self.edge_score_method,
                                      merge_method=self.merge_method
                                      )
                          )

        return torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, batch, n_id=None, dummy=False, x_null=None, e_null=None):
        """One forward pass applies the model over all steps.

        :param x: node features
        :param edge_index: list of edges
        :param batch: batching tensor
        :param n_id: optional global node ids from neighborloader. when provided,
            spotlights are initialised with these global ids so that subgraph lookups
            can be performed on the full graph instead of the sampled neighbourhood.
        """
        # fall back to local indices when no global mapping is given
        if n_id is None:
            n_id = torch.arange(len(x), device=x.device)

        merge_tree = {}
        spotlights = {}
        nodes = list(range(len(x)))
        merge_tree[0] = {n: set({}) for n in nodes}
        # level 0 spotlights hold the global id for each local node
        spotlights[0] = {n: {n_id[n].item()} for n in nodes}

        edge_index_initial = edge_index
        nodes_initial = nodes

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

            update_merge_graph(merge_tree, out['new_graph']['unpool'].cluster, t+1)
            update_spotlights(spotlights, out['new_graph']['unpool'].cluster, t+1)

            edge_index = out['new_graph']['e_ind_new']
            x = out['new_graph']['x_new']
            batch = out['new_graph']['batch_new']

        merge_info = {'tree': merge_tree, 'spotlights': spotlights}

        return xx, pp, ee, batches, merge_info, internals

    def rec_loss(self,
                 xx,
                 ee,
                 spotlights,
                 source_graph,
                 internals,
                 num_nodes=20,
                 draw=False):
        """Compute reconstruction loss at all coarsening levels.

        The loss function for a pair of embeddings z_1, z_2 and graph kernel K is:
        L = ((x_1 - x_2)^2  - K(g_1, g_2))^2
        where g_1 is the spotlight of node 1. Here we supervise the embedding for
        pairs of nodes.
        """
        # pull the full graph and its features once per call
        source_ig = source_graph.ig_graph
        source_x = source_graph.cached_data.x

        loss = 0
        for level in range(len(xx)):
            x = internals[level]['x_merged']

            # extract spotlight subgraphs from the full graph using global ids
            subgraphs, node_features = get_edge_subgraphs(ee[level],
                                                          spotlights,
                                                          level,
                                                          source_ig,
                                                          source_x,
                                                          None,
                                                          )

            K_predict = matrix_cosine(x[:num_nodes], x[:num_nodes])
            K_predict = K_predict.to(get_device())

            # wwl expects a list of per-graph float64 feature arrays
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
                        print('g1', x[i])
                        print('g2', x[j])
                        fig.suptitle(f"true: {K_true[i][j]}, pred: {K_predict[i][j]}")
                        plt.show()

            l = torch.nn.MSELoss()(K_predict, K_true)
            loss += l

        return loss / self.steps
    
    def rec_loss_wl(self,
                    xx,
                    ee,
                    spotlights,
                    source_graph,
                    internals,
                    num_nodes=20,
                    edge_sample_rate=1.0,
                    wl_iter=3,
                    ):
        """reconstruction loss using wl subtree kernel, vectorized.

        builds K_true via the batched wl subtree kernel: all spotlights
        at one pooling level are stacked into a single disjoint-union
        graph, wl labeling runs once on the union, and the full N x N
        similarity matrix is produced via histogram @ histogram.T per
        iteration.

        edge_sample_rate < 1.0 zeros a random subset of (i, j) entries
        in both K_predict and K_true so the loss only supervises a
        fraction of the pairs per batch.
        """
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

        loss = 0
        for level in range(len(xx)):
            x = internals[level]['x_merged']
            n_take = min(num_nodes, x.size(0))
            if n_take < 2:
                continue

            target_spotlights = spotlights.get(level + 1, spotlights.get(level, {}))

            edge_indices = []
            node_counts = []
            init_labels = []
            valid_local = []
            for spot_idx in range(n_take):
                spot_global_ids = target_spotlights.get(spot_idx)
                if not spot_global_ids:
                    continue
                node_idx = torch.tensor(
                    sorted(spot_global_ids), dtype=torch.long, device=device
                )
                ei_sub, _ = pyg_subgraph(
                    node_idx,
                    source_edge_index,
                    relabel_nodes=True,
                    num_nodes=n_source,
                )
                edge_indices.append(ei_sub)
                node_counts.append(node_idx.size(0))
                init_labels.append(source_labels[node_idx])
                valid_local.append(spot_idx)

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
    def distance_density(X, X_ref, k=20):
        """ Returns distance to kth nearest neighbor in batch """
        d = X.shape[1]
        N = X_ref.shape[0]
        knn = KDTree(X_ref.cpu().detach().numpy())
        R,_ = knn.query(X.cpu().detach().numpy(), k=k)
        R = R[:,k-1]
        return torch.tensor(R, dtype=torch.float32, requires_grad=False)

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

    def freq_loss(self,
                 internals_pos,
                 internals_neg,
                 pp,
                 estimator='knn',
                 beta=1,
                 lam=1,
                 steps=3,
                 volume=False,
                 k=30,
                 ):
        """ Penalize embeddings that are close to randos or sparse. """
        tot_loss = 0
        for t in range(len(pp)):
            x_pos = internals_pos[t]['x_merged']
            x_neg = internals_neg[t]['x_merged']
            s = pp[t]

            # cap k to the number of available reference points at this level
            k_eff = min(k, x_pos.size(0), x_neg.size(0))
            if k_eff < 2:
                continue

            if estimator == 'kde':
                density_pos = self.kde(x_pos, x_pos)
                density_neg = self.kde(x_pos, x_neg)
            if estimator == 'knn':
                density_pos = self.distance_density(x_pos, x_pos, k=k_eff)
                density_neg = self.distance_density(x_pos, x_neg, k=k_eff)

                # normalize to [0, 1] as the f_pos/f_neg comment below assumes
                scale = torch.cat([density_pos, density_neg]).max() + 1e-8
                density_pos = density_pos / scale
                density_neg = density_neg / scale
            if estimator == 'min':
                density_pos = self.min_density(x_pos, x_pos)
                density_neg = self.min_density(x_pos, x_neg)


            f_pos = density_pos.view(-1, 1).squeeze()
            f_neg  = density_neg.view(-1, 1).squeeze()

            # f_pos and f_neg are [0, 1]. When density is high f -> 0, 1 else
            # f_pos - f_neg -> -1 with motifs (f_p = 0, f_n = 1)
            # f_pos - f_neg -> 1 with non-motifs (f_p = 1, f_n=0)
            l = (-1 * s * torch.exp(-1 * beta * (f_pos - f_neg))).mean()

            reg_term = lam * s.pow(2.0).mean()
            l += reg_term

            tot_loss += l

        tot_loss /= steps
        return tot_loss

    def sil_loss(self, internals_pos, spotlights, tracker, momentum=0.95):
        """
        sampling invariance loss.

        neighborhood sampling exposes each node to varying local contexts across
        batches. a real motif instance should look the same regardless of which
        neighborhood it was sampled within. this loss tracks a momentum-updated
        target embedding for every (source_node, level) pair and penalises
        deviations of the current batch's embedding from the target.

        :param internals_pos: per-level internals returned by the forward pass
        :param spotlights: merge_info['spotlights'] for the current batch
        :param tracker: SamplingInvarianceTracker storing the targets
        :param momentum: smoothing factor for the target update (0.95 default)
        """
        device = get_device()
        tot_loss = torch.zeros(1, device=device)
        n_terms = 0

        for level in range(len(internals_pos)):
            x_level = internals_pos[level]['x_merged']
            if x_level.size(0) == 0:
                continue

            # normalize to compare direction rather than magnitude
            h_cur = F.normalize(x_level, dim=-1)

            # each supernode at this level represents one spotlight; we anchor the
            # target to the sorted spotlight tuple so it is stable across batches
            for node_idx in range(x_level.size(0)):
                spot = spotlights[level].get(node_idx)
                if not spot:
                    continue
                key = tuple(sorted(spot))

                target = tracker.get(key, level)
                if target is None:
                    # first sighting: seed the target with the current embedding
                    tracker.set(key, level, h_cur[node_idx].detach())
                    continue

                target = target.to(device)
                tot_loss = tot_loss + (1.0 - (h_cur[node_idx] * target).sum())
                n_terms += 1

                # ema update of the target; detach to keep it out of the graph
                new_target = momentum * target + (1.0 - momentum) * h_cur[node_idx].detach()
                new_target = F.normalize(new_target, dim=-1)
                tracker.set(key, level, new_target)

        if n_terms == 0:
            return torch.zeros(1, device=device).squeeze()
        return (tot_loss / n_terms).squeeze()


class SamplingInvarianceTracker:
    """
    stores momentum-updated target embeddings keyed by (spotlight, level).

    the spotlight (a tuple of original node ids) identifies a persistent motif
    candidate across batches even though neighborhood sampling varies the
    surrounding context each time. level is the contraction depth at which the
    embedding was produced.
    """
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
            pass

        embeddings = torch.stack(embeddings)
        return embeddings


def plot_K(K_true, K_pred):
    fig, ax = plt.subplots(1, 2)
    sns.heatmap(K_true.detach().numpy(), vmin=0, vmax=1, ax=ax[0])
    sns.heatmap(K_pred.detach().numpy(), vmin=0, vmax=1, ax=ax[1])
    plt.show()
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    pass
