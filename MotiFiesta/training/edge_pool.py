import random
from collections import namedtuple

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
        parallel_matching (bool, optional): if True, use the gpu-vectorized
            luby-style local-max matching instead of the sequential greedy
            matching. produces a different cluster assignment but preserves
            the stochastic-acceptance semantics of the original. default False.
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
                 parallel_matching=False,
                 ):
        super(EdgePooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv = torch_geometric.nn.GATConv(2 * in_channels, in_channels)
        self.conv_first = conv_first
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
        self.parallel_matching = parallel_matching

        dim = 2 if merge_method == 'cat' else 1
        # compute merged embeddings
        self.transform = torch.nn.Linear(dim * in_channels, out_channels)
        # self.transform_activate = torch.nn.ReLU()
        # self.transform_2 = torch.nn.Linear(out_channels, out_channels)
        # self.transform_activate = torch.nn.Sigmoid()

        # scoring layer
        self.score_net = torch.nn.Linear(out_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.score_net.reset_parameters()
        self.transform.reset_parameters()

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

    def forward(self, x, edge_index, batch, hard_embed=False, dummy=False):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        # carlos: do one conv operation before scoring
        x = x.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # computes features for each edge-connected node pair, normally just
        # sum pool

        # firt transform merged embeddings
        # print("x in ", x[:10])
        # print("x in ", x.shape)
        # print("in ", x)
        x_merged = self.edge_merge(x, edge_index)
        # print("in merge", x_merged)
        # print("x merge ", x_merged[:10])
        x_merged = self.transform(x_merged)
        # x_merged = self.transform_activate(x_merged)
        # x_merged = self.transform_2(x_merged)
        # print("x trans", x_merged[:10])
        # x_merged = self.transform_activate(x_merged)
        # print("x activate", x_merged[:10])
        # print("x merge and transf ", x_merged.shape)

        # compute features for each node with itself in case node is not pooled
        e_ind_self = torch.tensor([list(range(len(x))), list(range(len(x)))])
        x_merged_self = self.edge_merge(x, e_ind_self)
        x_merged_self = self.transform(x_merged_self)
        # x_merged_self = self.transform_activate(x_merged_self)
        # x_merged_self = self.transform_2(x_merged_self)

        # compute scores for each edge
        e = self.score_net(x_merged).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0), batch)

        if dummy:
            e = torch.full(e.shape, .5, dtype=torch.float32)

        if self.parallel_matching:
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

            # print(f"merged score: ", edge_score[edge_idx])
            merge_count += 1
            # emb_cat.append(torch.cat((x[source], x[target])))
            emb_cat.append(x_merged[edge_idx])
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

        # carlos
        new_x = torch.zeros((len(emb_cat), len(emb_cat[0])), dtype=torch.float)
        new_x = new_x.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
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

        # edge-free fast path: every node becomes a singleton
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

        # detect orientation: a symmetric (undirected) representation has
        # (v, u) present for every (u, v). if so, canonicalize to one
        # direction per pair. otherwise treat every edge as its own entry.
        edge_key_fwd = src_full * num_nodes + dst_full
        edge_key_rev = dst_full * num_nodes + src_full
        is_symmetric = torch.isin(edge_key_rev, edge_key_fwd).all().item()

        if is_symmetric:
            # undirected: src <= dst keeps one of each pair plus self-loops
            canon_mask = src_full <= dst_full
        else:
            # directed: each directed edge is its own canonical entry
            canon_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        canon_idx = canon_mask.nonzero(as_tuple=False).squeeze(-1)
        n_canon = canon_idx.size(0)

        # if for some reason every edge is non-canonical (shouldn't happen for
        # a well-formed undirected graph but defensive), all nodes become
        # singletons
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

        # random acceptance gate, matching the sequential version's behavior
        random_gate = torch.rand(n_canon, device=device)
        edges_alive = random_gate < canon_score

       # priority with small noise so per-node max is unique across ties.
        # use float64 to prevent collisions when sigmoid scores saturate near
        # 1.0 — float32 precision around 1.0 is ~6e-8, so 1e-7 noise can fail
        # to disambiguate ties, causing multiple selections to share endpoints
        # and corrupting cluster assignments
        priority = canon_score.double() + 1e-9 * torch.rand(
            n_canon, dtype=torch.float64, device=device
        )

        cluster = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        nodes_taken = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # store global edge indices (into full edge_index) for new_x assembly
        selected_per_round = []
        next_id = 0

        for _ in range(max_rounds):
            if not edges_alive.any():
                break

            # dead edges contribute -inf to the per-node max
            masked_priority = torch.where(
                edges_alive, priority,
                torch.full_like(priority, float('-inf'), dtype=torch.float64)
            )

            # max priority of incident alive edges at each node, considering
            # both src and dst incidences for undirected matching
            node_max = torch.full((num_nodes,), float('-inf'),
                                  dtype=torch.float64, device=device)
            node_max.scatter_reduce_(0, src, masked_priority,
                                     reduce='amax', include_self=True)
            node_max.scatter_reduce_(0, dst, masked_priority,
                                     reduce='amax', include_self=True)

            # an alive edge wins iff its priority is max at both endpoints
            # and neither endpoint has been taken in a previous round
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

            # any edge touching a now-taken node is dead going forward
            edges_alive = edges_alive & (~nodes_taken[src]) & (~nodes_taken[dst])

            # map local (canonical) indices back to global edge_index positions
            selected_per_round.append(canon_idx[sel_local])

        # remaining nodes become singletons with their own supernode id
        untaken = (~nodes_taken).nonzero(as_tuple=False).squeeze(-1)
        n_singletons = untaken.size(0)
        if n_singletons > 0:
            singleton_ids = torch.arange(next_id, next_id + n_singletons,
                                         device=device)
            cluster[untaken] = singleton_ids
            next_id += n_singletons

        total_nodes = next_id

        # assemble new_x: matched embeddings (in cluster id order), then singletons
        if selected_per_round:
            merged_idx_all = torch.cat(selected_per_round)
            merged_embeddings = x_merged[merged_idx_all]
        else:
            merged_idx_all = torch.empty(0, dtype=torch.long, device=device)
            merged_embeddings = torch.empty(0, x_merged.size(1), device=device)

        if n_singletons > 0:
            singleton_embeddings = x_merged_self[untaken]
        else:
            singleton_embeddings = torch.empty(
                0, x_merged_self.size(1), device=device)

        new_x = torch.cat([merged_embeddings, singleton_embeddings], dim=0)

        # new_edge_score: original score for merged edges, 1.0 for singletons
        if merged_idx_all.numel() > 0:
            merged_scores = edge_score[merged_idx_all]
        else:
            merged_scores = torch.empty(0, device=device)
        singleton_scores = (x.new_ones(n_singletons) if n_singletons > 0
                            else torch.empty(0, device=device))
        new_edge_score = torch.cat([merged_scores, singleton_scores])

        # coalesce edges into the new supernode graph, drop self loops
        N = total_nodes
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)
        new_edge_index, _ = remove_self_loops(new_edge_index)

        # batch id per new supernode: gather from any constituent node
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
    # quick sanity test: same input through both algorithms
    torch.manual_seed(0)

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
         [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]], dtype=torch.long)
    x = torch.randn(4, 3)
    batch = torch.zeros(4, dtype=torch.long)

    ep_g = EdgePooling(3, 2, edge_score_method='sigmoid', parallel_matching=False)
    ep_p = EdgePooling(3, 2, edge_score_method='sigmoid', parallel_matching=True)
    ep_p.load_state_dict(ep_g.state_dict())

    out_g = ep_g(x, edge_index, batch)
    out_p = ep_p(x, edge_index, batch)

    print("greedy cluster:  ", out_g['new_graph']['unpool'].cluster.tolist())
    print("parallel cluster:", out_p['new_graph']['unpool'].cluster.tolist())
    print("greedy x_new shape:  ", out_g['new_graph']['x_new'].shape)
    print("parallel x_new shape:", out_p['new_graph']['x_new'].shape)
