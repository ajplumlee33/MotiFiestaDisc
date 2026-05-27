import random
from collections import namedtuple

import networkx as nx
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import softmax
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_add_pool


class EdgePooling(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.

    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index",
         "cluster",
         "batch",
         "new_edge_score",
         "old_edge_score"])

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_score_method='sigmoid',
                 dropout=0,
                 merge_method='sum',
                 add_to_edge_score=0.0,
                 conv_first=False,
                 matching_mode='greedy',
                 scoring_mode='mlp',
                 n_heads=4,
                 ):
        super(EdgePooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_first = conv_first
        self.matching_mode = matching_mode
        self.scoring_mode = scoring_mode
        if edge_score_method == 'softmax':
            edge_score_method = self.compute_edge_score_softmax
        elif edge_score_method == 'sigmoid':
            edge_score_method = self.compute_edge_score_sigmoid
        else:
            edge_score_method = self.compute_edge_score_softmax_full

        if merge_method == 'cat':
            self.edge_merge = self.merge_edge_cat
        else:
            self.edge_merge = self.merge_edge_sum

        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout
        self.merge_method = merge_method

        dim = 2 if merge_method == 'cat' else 1
        self.transform = torch.nn.Linear(dim * in_channels, out_channels)

        if scoring_mode == 'attention':
            self.n_heads = n_heads
            self.head_dim = max(1, in_channels // n_heads)
            self.attn_q = torch.nn.Linear(in_channels, self.head_dim * n_heads, bias=False)
            self.attn_k = torch.nn.Linear(in_channels, self.head_dim * n_heads, bias=False)
            self.attn_out = torch.nn.Linear(n_heads, 1, bias=False)
        else:
            # +3: cn, degree product, k-core product (structural signals, score path only)
            score_in = out_channels + 3
            self.score_net = torch.nn.Sequential(
                torch.nn.Linear(score_in, score_in),
                torch.nn.ReLU(),
                torch.nn.Linear(score_in, 1),
            )

        self.reset_parameters()

    def reset_parameters(self):
        self.transform.reset_parameters()
        if self.scoring_mode == 'attention':
            self.attn_q.reset_parameters()
            self.attn_k.reset_parameters()
            self.attn_out.reset_parameters()
        else:
            for m in self.score_net.modules():
                if isinstance(m, torch.nn.Linear):
                    m.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes, batch):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_softmax_full(raw_edge_score, edge_index, num_nodes, batch):
        e_batch = batch[edge_index[0]]
        return softmax(raw_edge_score, e_batch)

    @staticmethod
    def compute_edge_score_dummy(raw_edge_score, edge_index, num_nodes, batch):
        return torch.tensor([.5] * edge_index.shape[1], dtype=torch.float)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes, batch):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes, batch):
        return torch.sigmoid(raw_edge_score)
    @staticmethod
    def merge_edge_cat(x, edge_index):
        return torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)

    @staticmethod
    def merge_edge_sum(x, edge_index):
        X = x[torch.flatten(edge_index.T)]
        batch = torch.arange(0, len(edge_index[0])).repeat_interleave(2)
        batch = batch.to(X.device)
        return global_add_pool(X, batch)

    def forward(self, x, edge_index, batch, hard_embed=False, dummy=False, x_score=None):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features used for the hash embedding (transform path).
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.
            x_score (Tensor, optional): If provided, used instead of x for
                score_net input (dual-pathway: separate scoring and embedding streams).

        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        raw = self.edge_merge(x, edge_index)
        x_merged = self.transform(raw)

        e_ind_self = torch.tensor([list(range(len(x))), list(range(len(x)))], device=x.device)
        raw_self = self.edge_merge(x, e_ind_self)
        x_merged_self = self.transform(raw_self)

        if self.scoring_mode == 'attention':
            x_attn = x_score if x_score is not None else x
            q = self.attn_q(x_attn).view(x_attn.size(0), self.n_heads, self.head_dim)
            k = self.attn_k(x_attn).view(x_attn.size(0), self.n_heads, self.head_dim)
            q_src = q[edge_index[0]]
            k_dst = k[edge_index[1]]
            attn = (q_src * k_dst).sum(dim=-1) / (self.head_dim ** 0.5)
            e = self.attn_out(attn).squeeze(-1)
        else:
            with torch.no_grad():
                num_nodes = x.size(0)
                adj = torch.zeros(num_nodes, num_nodes, device=x.device)
                adj[edge_index[0], edge_index[1]] = 1.0
                # common-neighbor count: high for clique edges, near-0 for sparse background
                cn = (adj @ adj)[edge_index[0], edge_index[1]].unsqueeze(-1)
                cn = torch.log1p(cn)
                # degree product: high when both endpoints are hubs (clique nodes)
                deg = adj.sum(dim=1)
                dp = (deg[edge_index[0]] * deg[edge_index[1]]).unsqueeze(-1)
                dp = torch.log1p(dp)
                # k-core number per node: 9-core for K10 clique nodes, low for background
                ei_cpu = edge_index.cpu()
                G = nx.Graph()
                G.add_nodes_from(range(x.size(0)))
                G.add_edges_from(zip(ei_cpu[0].tolist(), ei_cpu[1].tolist()))
                core = nx.core_number(G)
                core_t = torch.tensor([core[i] for i in range(x.size(0))],
                                      dtype=torch.float32, device=x.device)
                kc = (core_t[edge_index[0]] * core_t[edge_index[1]]).unsqueeze(-1)
                kc = torch.log1p(kc)
            # dual pathway: use gin-enriched edge features for scoring if provided
            if x_score is not None:
                raw_score = self.edge_merge(x_score, edge_index)
                x_for_score = raw_score
            else:
                x_for_score = x_merged
            e = self.score_net(torch.cat([F.relu(x_for_score), cn, dp, kc], dim=-1)).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0), batch)

        if dummy:
            e = torch.full(e.shape, .5, dtype=torch.float32, device=e.device)

        if self.matching_mode == 'luby':
            x_new, edge_index, batch, unpool_info = self.__merge_edges_parallel__(
                x, edge_index, batch, e, x_merged, x_merged_self)
        else:
            x_new, edge_index, batch, unpool_info = self.__merge_edges__(
                x, edge_index, batch, e, x_merged, x_merged_self)

        return {'new_graph': {'x_new': x_new, 'e_ind_new': edge_index, 'batch_new': batch, 'unpool': unpool_info},
                'internals': {'x_merged': x_merged, 'x_merged_self': x_merged_self, 'edge_scores': e}
                }

    def __merge_edges__(self, x, edge_index, batch, edge_score, x_merged, x_merged_self):
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []

        # carlos : start building the new x
        emb_cat = []

        # edge_score_norm = torch.nn.Softmax()(edge_score.detach())

        merge_count = 0

        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            # carlos
            r = random.random()
            # if r > edge_score[edge_idx] - .5:
            if r > edge_score[edge_idx]:
            # if r > edge_score_norm[edge_idx]:
                # print("skipped ", edge_score[edge_idx])
                continue

            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            merge_count += 1
            e_sc = edge_score[edge_idx]
            ste_gate = e_sc + (1.0 - e_sc).detach()
            emb_cat.append(x_merged[edge_idx] * ste_gate)
            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            # emb_cat.append(torch.cat((x[node_idx], x[node_idx])))
            emb_cat.append(x_merged_self[node_idx])
            i += 1

        cluster = cluster.to(x.device)

        new_x = torch.zeros((len(emb_cat), len(emb_cat[0])), dtype=torch.float, device=x.device)
        for ind, emb in enumerate(emb_cat):
            new_x[ind] = emb

        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])

        # scale embedding with score
        # might want to take this out since we use the embedding in
        # reconstruction
        # new_x = self.squish(new_x)
        # new_x = new_x * new_edge_score.view(-1, 1)

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)
        # I added this.. for some reason we were creating self loops..
        new_edge_index,_ = remove_self_loops(new_edge_index)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        # i added e to the output (edge scores for original graph)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score,
                                              old_edge_score=edge_score)

        return new_x, new_edge_index, new_batch, unpool_info

    def __merge_edges_parallel__(self, x, edge_index, batch, edge_score,
                                  x_merged, x_merged_self, max_rounds=20):
        """gpu-vectorized stochastic maximal matching via luby-style selection.

        each edge passes a random gate proportional to its score, matching the
        sequential version's `if r > edge_score: continue` semantics. surviving
        edges enter parallel matching: per round, an edge is selected iff its
        priority (score plus tiny tiebreaker noise) is the max among alive
        edges at both its endpoints. converges in roughly o(log n) rounds.

        canonicalization: pyg represents undirected edges as both (u, v) and
        (v, u) columns. with symmetric merge methods (sum), both directions
        have identical scores; float-precision noise can fail to distinguish
        them, causing both to be selected and consuming two supernode ids per
        undirected edge. we filter to src <= dst before matching so each
        undirected edge can be selected at most once. the full edge_index is
        still used to construct the new graph via cluster[edge_index].
        """
        device = x.device
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            cluster = torch.arange(num_nodes, device=device, dtype=torch.long)
            new_x = x_merged_self
            new_edge_score = x.new_ones(num_nodes)
            new_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            new_batch = batch.clone()
            unpool_info = self.unpool_description(
                edge_index=edge_index, cluster=cluster, batch=batch,
                new_edge_score=new_edge_score, old_edge_score=edge_score,
            )
            return new_x, new_edge_index, new_batch, unpool_info

        src_full = edge_index[0]
        dst_full = edge_index[1]

        edge_key_fwd = src_full * num_nodes + dst_full
        edge_key_rev = dst_full * num_nodes + src_full
        is_symmetric = torch.isin(edge_key_rev, edge_key_fwd).all().item()

        if is_symmetric:
            canon_mask = src_full <= dst_full
        else:
            canon_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        canon_idx = canon_mask.nonzero(as_tuple=False).squeeze(-1)
        n_canon = canon_idx.size(0)

        if n_canon == 0:
            cluster = torch.arange(num_nodes, device=device, dtype=torch.long)
            new_x = x_merged_self
            new_edge_score = x.new_ones(num_nodes)
            new_edge_index, _ = coalesce(cluster[edge_index], None, num_nodes, num_nodes)
            new_edge_index, _ = remove_self_loops(new_edge_index)
            new_batch = batch.clone()
            unpool_info = self.unpool_description(
                edge_index=edge_index, cluster=cluster, batch=batch,
                new_edge_score=new_edge_score, old_edge_score=edge_score,
            )
            return new_x, new_edge_index, new_batch, unpool_info

        src = src_full[canon_idx]
        dst = dst_full[canon_idx]
        canon_score = edge_score[canon_idx]

        random_gate = torch.rand(n_canon, device=device)
        edges_alive = random_gate < canon_score

        if device.type == 'mps':
            # mps lacks float64; use float32 with larger noise for tiebreaking
            priority = canon_score + 1e-4 * torch.rand(n_canon, dtype=torch.float32, device=device)
        else:
            priority = canon_score.double() + 1e-9 * torch.rand(n_canon, dtype=torch.float64, device=device)

        cluster = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        nodes_taken = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        selected_per_round = []
        next_id = 0

        for _ in range(max_rounds):
            if not edges_alive.any():
                break

            masked_priority = torch.where(
                edges_alive, priority,
                torch.full_like(priority, float('-inf'))
            )

            if device.type == 'mps':
                # scatter_reduce_ amax unsupported on mps; num_nodes is tiny so cpu cost is negligible
                _nm = torch.full((num_nodes,), float('-inf'))
                _nm.scatter_reduce_(0, src.cpu(), masked_priority.cpu(),
                                    reduce='amax', include_self=True)
                _nm.scatter_reduce_(0, dst.cpu(), masked_priority.cpu(),
                                    reduce='amax', include_self=True)
                node_max = _nm.to(device)
            else:
                node_max = torch.full((num_nodes,), float('-inf'),
                                      dtype=torch.float64, device=device)
                node_max.scatter_reduce_(0, src, masked_priority,
                                         reduce='amax', include_self=True)
                node_max.scatter_reduce_(0, dst, masked_priority,
                                         reduce='amax', include_self=True)

            selected = (
                edges_alive
                & (masked_priority == node_max[src])
                & (masked_priority == node_max[dst])
                & (~nodes_taken[src])
                & (~nodes_taken[dst])
            )

            if not selected.any():
                break

            sel_local = selected.nonzero(as_tuple=False).squeeze(-1)
            n_new = sel_local.size(0)
            new_ids = torch.arange(next_id, next_id + n_new, device=device)
            next_id += n_new

            sel_src = src[sel_local]
            sel_dst = dst[sel_local]

            cluster[sel_src] = new_ids
            cluster[sel_dst] = new_ids

            nodes_taken[sel_src] = True
            nodes_taken[sel_dst] = True

            edges_alive = edges_alive & (~nodes_taken[src]) & (~nodes_taken[dst])
            selected_per_round.append(canon_idx[sel_local])

        untaken = (~nodes_taken).nonzero(as_tuple=False).squeeze(-1)
        n_singletons = untaken.size(0)
        if n_singletons > 0:
            singleton_ids = torch.arange(next_id, next_id + n_singletons, device=device)
            cluster[untaken] = singleton_ids
            next_id += n_singletons

        total_nodes = next_id

        if selected_per_round:
            merged_idx_all = torch.cat(selected_per_round)
            ste_scores = edge_score[merged_idx_all]
            ste_gate = ste_scores + (1.0 - ste_scores).detach()
            merged_embeddings = x_merged[merged_idx_all] * ste_gate.unsqueeze(-1)
        else:
            merged_idx_all = torch.empty(0, dtype=torch.long, device=device)
            merged_embeddings = torch.empty(0, x_merged.size(1), device=device)

        if n_singletons > 0:
            singleton_embeddings = x_merged_self[untaken]
        else:
            singleton_embeddings = torch.empty(0, x_merged_self.size(1), device=device)

        new_x = torch.cat([merged_embeddings, singleton_embeddings], dim=0)

        if merged_idx_all.numel() > 0:
            merged_scores = edge_score[merged_idx_all]
        else:
            merged_scores = torch.empty(0, device=device)
        singleton_scores = (x.new_ones(n_singletons) if n_singletons > 0
                            else torch.empty(0, device=device))
        new_edge_score = torch.cat([merged_scores, singleton_scores])

        N = total_nodes
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)
        new_edge_index, _ = remove_self_loops(new_edge_index)

        new_batch = torch.empty(N, dtype=torch.long, device=device)
        new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(
            edge_index=edge_index, cluster=cluster, batch=batch,
            new_edge_score=new_edge_score, old_edge_score=edge_score,
        )

        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.

        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """

        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.in_channels)


if __name__ == "__main__":
    ep = EdgePooling(3, 2, edge_score_method='sigmoid')
    edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1, 0, 1], [0, 2, -1], [-1, 0, 1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    o = ep(data.x, data.edge_index, torch.tensor([0] * len(x), dtype=torch.long))
    pass
