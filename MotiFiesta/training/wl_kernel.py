"""wl subtree kernel for graph similarity, pure torch.

drop-in replacement for the wwl kernel in rec_loss. operates on
edge_index tensors and integer label tensors. runs on whatever device
the inputs live on. no networkx, no scipy, no POT.

reference: shervashidze et al., 'weisfeiler-lehman graph kernels',
JMLR 2011. specifically the subtree variant (algorithm 2).

two entry points:
    wl_subtree_similarity(...)        : pairwise, single similarity scalar.
    wl_subtree_similarity_batch(...)  : N graphs in, N x N matrix out.
                                        vectorized via disjoint-union stacking.
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


# ---------------------------------------------------------------------------
# pairwise api (single similarity)
# ---------------------------------------------------------------------------

def wl_subtree_similarity(
    edge_index_a, num_nodes_a, init_labels_a,
    edge_index_b, num_nodes_b, init_labels_b,
    n_iter=3,
    normalize=True,
):
    """compute wl subtree kernel similarity between two graphs.

    args:
        edge_index_{a,b}: long tensor of shape (2, E). assumed undirected
            (both directions present in edge_index).
        num_nodes_{a,b}: int.
        init_labels_{a,b}: long tensor of shape (num_nodes,). initial
            integer label per node.
        n_iter: number of wl relabeling iterations.
        normalize: if True, return cosine-normalized similarity in [0, 1].

    returns:
        scalar tensor on the same device as inputs.
    """
    K = wl_subtree_similarity_batch(
        [edge_index_a, edge_index_b],
        [num_nodes_a, num_nodes_b],
        [init_labels_a, init_labels_b],
        n_iter=n_iter,
        normalize=normalize,
    )
    return K[0, 1]


# ---------------------------------------------------------------------------
# batched api (full pairwise matrix)
# ---------------------------------------------------------------------------

def wl_subtree_similarity_batch(
    edge_indices,
    node_counts,
    init_labels,
    n_iter=3,
    normalize=True,
):
    """compute pairwise wl subtree kernel matrix for a batch of graphs.

    works by stacking the N graphs into one disjoint-union graph (offsets
    applied to node ids so each graph is a disconnected component), then
    running wl relabeling once on the union. labels at each iteration are
    in a shared canonical space across all graphs, so per-graph histograms
    can be combined via histogram @ histogram.T to produce the full N x N
    similarity matrix.

    args:
        edge_indices: list of N long tensors of shape (2, E_i). each
            edge_index uses local node ids in [0, N_i).
        node_counts: list of N ints, num_nodes per graph.
        init_labels: list of N long tensors of shape (N_i,).
        n_iter: number of wl iterations.
        normalize: if True, return cosine-normalized N x N matrix.

    returns:
        N x N similarity matrix on the same device as inputs.
    """
    n_graphs = len(node_counts)
    if n_graphs == 0:
        raise ValueError("empty input")
    device = init_labels[0].device

    # stack into one disjoint-union graph: each graph's node ids get
    # offset by the cumulative node count of preceding graphs. since
    # no cross-graph edges are added, wl labeling on the union is
    # identical to running it independently per graph, but produces
    # labels in a single shared label space.
    offsets = torch.zeros(n_graphs + 1, dtype=torch.long, device=device)
    for i, n in enumerate(node_counts):
        offsets[i + 1] = offsets[i] + n
    total_nodes = int(offsets[-1].item())

    # stacked edge_index with offsets
    pieces = []
    for ei, off in zip(edge_indices, offsets[:-1].tolist()):
        if ei.numel() > 0:
            pieces.append(ei + off)
    if pieces:
        stacked_ei = torch.cat(pieces, dim=1)
    else:
        stacked_ei = torch.zeros(2, 0, dtype=torch.long, device=device)

    # graph_id per node: which graph each node in the union belongs to
    graph_id = torch.cat([
        torch.full((n,), i, dtype=torch.long, device=device)
        for i, n in enumerate(node_counts)
    ])

    # initial labels remapped to a joint canonical space
    stacked_labels = torch.cat(init_labels)
    _, labels = torch.unique(stacked_labels, return_inverse=True)

    K = torch.zeros(n_graphs, n_graphs, device=device)

    for it in range(n_iter + 1):
        # build per-graph histograms over the current joint label space
        n_labels = int(labels.max().item()) + 1
        labels_onehot = F.one_hot(labels, num_classes=n_labels).float()
        # segment-sum into per-graph rows: (n_graphs, n_labels)
        hist = scatter(
            labels_onehot, graph_id, dim=0,
            dim_size=n_graphs, reduce='sum',
        )
        # K[i, j] += <hist[i], hist[j]>; full N x N in one matmul
        K = K + hist @ hist.T

        if it < n_iter:
            labels = _wl_relabel_union(labels, stacked_ei, total_nodes, n_labels)

    if normalize:
        diag = K.diag().clamp(min=1e-12).sqrt()
        K = K / (diag.unsqueeze(0) * diag.unsqueeze(1))

    return K


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _wl_relabel_union(labels, edge_index, num_nodes, n_labels):
    """one wl relabel step on a (possibly disjoint-union) graph.

    each node's new label = canonical id of (own_label, multiset of
    neighbor labels). multiset is encoded as a one-hot histogram for
    permutation invariance. canonical ids come from torch.unique on
    the full signature matrix, so labels stay in a shared space.
    """
    if edge_index.size(1) == 0:
        signature = labels.unsqueeze(1)
    else:
        src, dst = edge_index[0], edge_index[1]
        src_onehot = F.one_hot(labels[src], num_classes=n_labels).long()
        neighbor_hist = scatter(
            src_onehot, dst, dim=0,
            dim_size=num_nodes, reduce='sum',
        )
        signature = torch.cat([labels.unsqueeze(1), neighbor_hist], dim=1)
    _, new_labels = torch.unique(signature, dim=0, return_inverse=True)
    return new_labels


# ---------------------------------------------------------------------------
# convenience wrappers
# ---------------------------------------------------------------------------

def initial_labels_from_onehot(x):
    """convert one-hot (e.g. onehotdegree) features to integer labels.

    args:
        x: float tensor of shape (N, F), expected to be one-hot per row.

    returns:
        long tensor of shape (N,).
    """
    return x.argmax(dim=-1)


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # complete graph on four nodes
    edges_k4 = torch.tensor([
        [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
        [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2],
    ])
    labels_k4 = torch.zeros(4, dtype=torch.long)

    # path of four nodes
    edges_p4 = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    labels_p4 = torch.zeros(4, dtype=torch.long)

    # cycle of four nodes
    edges_c4 = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
    ])
    labels_c4 = torch.zeros(4, dtype=torch.long)

    # ----- pairwise sanity checks -----
    print("pairwise:")
    print(f"  K(K4, K4) = {wl_subtree_similarity(edges_k4, 4, labels_k4, edges_k4, 4, labels_k4, n_iter=3).item():.4f}")
    print(f"  K(P4, P4) = {wl_subtree_similarity(edges_p4, 4, labels_p4, edges_p4, 4, labels_p4, n_iter=3).item():.4f}")
    print(f"  K(K4, P4) = {wl_subtree_similarity(edges_k4, 4, labels_k4, edges_p4, 4, labels_p4, n_iter=3).item():.4f}")
    print(f"  K(C4, P4) = {wl_subtree_similarity(edges_c4, 4, labels_c4, edges_p4, 4, labels_p4, n_iter=3).item():.4f}")

    # ----- batched should agree with pairwise -----
    print("\nbatched matrix for [K4, P4, C4]:")
    K = wl_subtree_similarity_batch(
        [edges_k4, edges_p4, edges_c4],
        [4, 4, 4],
        [labels_k4, labels_p4, labels_c4],
        n_iter=3,
    )
    print(K)

    pair_kp = wl_subtree_similarity(edges_k4, 4, labels_k4, edges_p4, 4, labels_p4, n_iter=3).item()
    batch_kp = K[0, 1].item()
    print(f"\npairwise K(K4, P4) = {pair_kp:.6f}")
    print(f"batched  K[0, 1]   = {batch_kp:.6f}")
    print(f"agree to 1e-5: {abs(pair_kp - batch_kp) < 1e-5}")

    # ----- speed comparison on a realistic-ish batch -----
    print("\nspeed: 20 small random graphs (simulates one pool level):")
    n_graphs = 20
    rng = torch.Generator().manual_seed(0)

    edge_list, count_list, label_list = [], [], []
    for _ in range(n_graphs):
        n = int(torch.randint(5, 15, (1,), generator=rng).item())
        e = max(1, n // 2)
        src = torch.randint(0, n, (e,), generator=rng)
        dst = torch.randint(0, n, (e,), generator=rng)
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        ei = ei[:, ei[0] != ei[1]]
        edge_list.append(ei)
        count_list.append(n)
        label_list.append(torch.randint(0, 5, (n,), generator=rng))

    t0 = time.time()
    K_pair = torch.zeros(n_graphs, n_graphs)
    for i in range(n_graphs):
        for j in range(i, n_graphs):
            k = wl_subtree_similarity(
                edge_list[i], count_list[i], label_list[i],
                edge_list[j], count_list[j], label_list[j],
                n_iter=3,
            )
            K_pair[i, j] = k
            K_pair[j, i] = k
    t_pair = time.time() - t0

    t0 = time.time()
    K_batch = wl_subtree_similarity_batch(
        edge_list, count_list, label_list, n_iter=3,
    )
    t_batch = time.time() - t0

    print(f"  pairwise: {t_pair*1000:.1f} ms ({n_graphs * (n_graphs+1) // 2} kernel calls)")
    print(f"  batched : {t_batch*1000:.1f} ms (1 kernel call)")
    print(f"  speedup : {t_pair / t_batch:.1f}x")
    print(f"  matrices match (1e-4 tol): {(K_pair - K_batch).abs().max().item() < 1e-4}")

    if torch.cuda.is_available():
        edge_list_gpu = [e.cuda() for e in edge_list]
        label_list_gpu = [l.cuda() for l in label_list]
        K_gpu = wl_subtree_similarity_batch(
            edge_list_gpu, count_list, label_list_gpu, n_iter=3,
        )
        print(f"\ngpu batched output device: {K_gpu.device}")
