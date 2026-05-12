import itertools
from collections import defaultdict

import torch
from torch_geometric.data import Data
import networkx as nx
from networkx.algorithms.swap import connected_double_edge_swap
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.utils import k_hop_subgraph


def induced_edge_filter_(G, roots, depth=1):
    """Remove edges in G introduced by the induced subgraph routine.

    Only keep edges which fall within a single node's neighbourhood.
    """
    if depth < 1:
        depth = 1
    neighbourhoods = []
    flat_neighbors = set()
    for root in roots:
        root_neighbors = bfs_expand(G, [root], hops=depth)
        neighbourhoods.append(root_neighbors)
        flat_neighbors = flat_neighbors.union(root_neighbors)

    flat_neighbors = list(flat_neighbors)
    subG = G.subgraph(flat_neighbors).copy()
    kill = []
    for (u, v) in subG.edges():
        for nei in neighbourhoods:
            if u in nei and v in nei:
                break
        else:
            kill.append((u, v))

    subG.remove_edges_from(kill)
    return subG


def induced_edge_filter(G, roots):
    kill = []
    for (u, v) in G.edges():
        if u not in roots and v not in roots:
            kill.append((u, v))
    G.remove_edges_from(kill)


def bfs_expand(G, initial_nodes, hops=2):
    total_nodes = [list(initial_nodes)]
    for d in range(hops):
        depth_ring = []
        for n in total_nodes[d]:
            for nei in G.neighbors(n):
                depth_ring.append(nei)
        total_nodes.append(depth_ring)
    return set(itertools.chain(*total_nodes))


def bfs(G, initial_node, depth=2):
    total_nodes = [[initial_node]]
    visited = []
    for d in range(depth):
        depth_ring = []
        for n in total_nodes[d]:
            visited.append(n)
            for nei in G.neighbors(n):
                if nei not in visited:
                    depth_ring.append(nei)
        total_nodes.append(depth_ring)
        yield depth_ring


def to_graphs(batch):
    """Convert batch to list of networkx subgraphs."""
    big_g = to_networkx(batch)
    nodelist = lambda i: list(np.argwhere(batch.batch == i).numpy()[0])
    graphs = [big_g.subgraph(nodelist(i)).copy()
              for i in range(batch.num_graphs)]
    return graphs


def _is_single_source_graph(graphs):
    """single-graph mode passes one igraph.Graph instead of a list of nx graphs."""
    from igraph import Graph as IGraph
    return isinstance(graphs, IGraph)


def get_edge_subgraphs(edge_index, spotlights, level, graphs, x_base, batch, hop=False):
    """one spotlight subgraph per edge, dual-mode."""
    single = _is_single_source_graph(graphs)

    subgraphs = []
    X = []
    for u, v in edge_index.T:
        spotlight_u = spotlights[level][u.item()]
        spotlight_v = spotlights[level][v.item()]
        spotlight = spotlight_u.union(spotlight_v)

        if single:
            subgraph = graphs.subgraph(sorted(spotlight))
        else:
            graph = graphs[batch[list(spotlight)[0]]]
            subgraph = graph.subgraph(spotlight).copy()

        node_features = np.stack([x_base[n].cpu().numpy() for n in sorted(list(spotlight))])
        subgraphs.append(subgraph)
        X.append(node_features)

    return subgraphs, X


def get_subgraphs(node_ids, spotlights, level, graphs, x_base, batch, hop=False):
    """one spotlight subgraph per node_id, dual-mode."""
    single = _is_single_source_graph(graphs)

    subgraphs = []
    X = []
    for node in node_ids:
        spotlight = spotlights[level][node]

        if single:
            subgraph = graphs.subgraph(sorted(spotlight))
        else:
            graph = graphs[batch[list(spotlight)[0]]]
            subgraph = graph.subgraph(spotlight).copy()

        node_features = np.stack([x_base[n].cpu().numpy() for n in sorted(list(spotlight))])
        subgraphs.append(subgraph)
        X.append(node_features)

    return subgraphs, X


def get_subgraph_edge(u, v, spotlights, level, graphs, batch, hop=False):
    """get spotlight from contracting edge (u, v)."""
    u, v = u.item(), v.item()
    spotlight_u = spotlights[level][u]
    spotlight_v = spotlights[level][v]
    spotlight_uv = spotlight_u | spotlight_v

    n = list(spotlight_uv)[0]
    graph = graphs[batch[n]]

    if hop:
        spotlight_uv = bfs_expand(graph, spotlight_uv, hops=1)

    return graph.subgraph(spotlight_uv).copy()


def expand_spotlights(spotlights, t, edge_index, k):
    """merge k-hop neighbourhood spotlights."""
    if k < 1:
        return
    nodes = range(len(spotlights[t]))
    new_spotlights = defaultdict(set)
    for n in nodes:
        if len(edge_index[0]) == 0:
            continue
        nei = k_hop_subgraph(n, k, edge_index)
        neis = nei[0]
        new_nodes = set()
        for u in neis:
            new_nodes |= spotlights[t][u.item()]
        new_spotlights[n] = new_nodes | {n}

    for n, sp in new_spotlights.items():
        spotlights[t][n] = sp


def update_spotlights(spotlights, clusters, t):
    """keeps track of the spotlight of each node.

    >>> from collections import defaultdict
    >>> import torch
    >>> SL = {0: {0: {1, 2}, 1: {3, 4} }}
    >>> clusters = torch.tensor([0, 0], dtype=torch.long)
    >>> update_spotlights(SL, clusters, 1)
    >>> SL
    {0: {0: {1, 2}, 1: {3, 4}}, 1: defaultdict(<class 'set'>, {0: {1, 2, 3, 4}})}
    """
    spotlights[t] = defaultdict(set)
    for i, c in enumerate(clusters):
        spotlights[t][c.item()] |= spotlights[t-1][i]


def update_merge_graph(merge_graph, clusters, t):
    """keeps track of the children of each node."""
    merge_graph[t] = defaultdict(set)
    for i, c in enumerate(clusters):
        merge_graph[t][c.item()] |= {i}


def draw_one_instance(g_data, spotlight, show=False):
    G = to_networkx(g_data)
    nx.draw(G)
    if show:
        plt.show()


def ablate_graphs(graphs, method='swap', n_swaps=5):
    """take a batch of graphs and perform an ablation meant to be used
    as the 'configuration' model. legacy networkx-based path; for new
    single-graph rewiring use torch_double_edge_swap instead.
    """
    graphs_swap = []
    for g in graphs:
        graph_swap = g.copy().to_undirected()
        connected_double_edge_swap(graph_swap, nswap=n_swaps)
        graphs_swap.append(graph_swap.to_directed())

    return graphs_swap


def batch_to_node_indices(batch):
    """return node indices within each graph for a given batch.

    >>> import torch
    >>> batch = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    >>> batch_to_node_indices(batch)
    [0, 1, 2, 0, 1, 0]
    """
    assert bool((batch == torch.sort(batch)[0]).all()), "batch indices not sorted"
    indices = [0]
    current_batch = batch[0]
    ind = 1
    for b in batch[1:]:
        if b != current_batch:
            ind = 0
            current_batch = b
        indices.append(ind)
        ind += 1
    assert len(indices) == len(batch)
    return indices


# ---------------------------------------------------------------------------
# tensor-based spotlight tracking
# ---------------------------------------------------------------------------
# spotlight_assignment[t]: long tensor (n_batch_nodes,) where entry i is the
# supernode index at level t that original local node i belongs to.
# cluster_chain[t]: long tensor (n_supernodes_at_t,), maps level-t supernode
# index to level-(t+1) supernode index.
# n_id: long tensor (n_batch_nodes,) of global node ids.


def spotlight_at(spotlight_assignment, n_id, t, supernode_idx):
    """global node ids in the spotlight of supernode `supernode_idx` at level t."""
    mask = spotlight_assignment[t] == supernode_idx
    local_members = mask.nonzero(as_tuple=False).squeeze(-1)
    return n_id[local_members]


def edge_spotlight(spotlight_assignment, n_id, t, u_idx, v_idx):
    """global node ids in the union of u's and v's spotlights at level t."""
    spot_t = spotlight_assignment[t]
    mask = (spot_t == u_idx) | (spot_t == v_idx)
    local_members = mask.nonzero(as_tuple=False).squeeze(-1)
    return n_id[local_members]


def children_at(cluster_chain, t, supernode_idx):
    """level-(t-1) supernode indices that merged into `supernode_idx` at level t."""
    if t < 1:
        return torch.empty(0, dtype=torch.long)
    cluster = cluster_chain[t - 1]
    mask = cluster == supernode_idx
    return mask.nonzero(as_tuple=False).squeeze(-1)


def spotlight_key(global_ids):
    """hashable key (sorted tuple of ints) for tracker lookups."""
    if isinstance(global_ids, torch.Tensor):
        return tuple(sorted(global_ids.tolist()))
    return tuple(sorted(global_ids))


def get_edge_subgraphs_tensor(edge_index, spotlight_assignment, n_id, level,
                              source_graph, source_x):
    """edge spotlight subgraphs from the tensor representation.

    for each edge (u, v) in edge_index, builds the induced subgraph of
    source_graph over the union of u's and v's spotlights at level. returns
    (list of igraph subgraphs, list of feature arrays).
    """
    subgraphs = []
    X = []
    spot_t = spotlight_assignment[level]
    n_id_cpu = n_id.cpu()

    for u, v in edge_index.T:
        u_idx = u.item()
        v_idx = v.item()
        mask = (spot_t == u_idx) | (spot_t == v_idx)
        local_members = mask.nonzero(as_tuple=False).squeeze(-1).cpu()
        global_ids_sorted = sorted(n_id_cpu[local_members].tolist())

        subgraph = source_graph.subgraph(global_ids_sorted)
        node_features = np.stack([
            source_x[g].cpu().numpy() for g in global_ids_sorted
        ])

        subgraphs.append(subgraph)
        X.append(node_features)

    return subgraphs, X


# ---------------------------------------------------------------------------
# negative sample generation
# ---------------------------------------------------------------------------

def torch_double_edge_swap(edge_index, n_swaps=20):
    """parallel double edge swap for negative sample generation.

    classical double edge swap: pick two undirected edges (a, b) and (c, d),
    swap to (a, d) and (b, c). networkx does this sequentially with a
    connectivity-preservation check; here we pick 2*n_swaps disjoint
    canonical edges, pair them up, and apply all swaps simultaneously.

    drops the connectivity constraint - fine for negative samples used in
    contrastive losses where the goal is structural randomization, not
    preservation of a single connected component. preserves degree sequence
    exactly because each swap only rearranges existing endpoints.

    operates on the canonical (src < dst) representation and returns the
    symmetric (both-direction) edge_index that pyg expects.
    """
    device = edge_index.device
    src_full, dst_full = edge_index[0], edge_index[1]

    canon = src_full < dst_full
    can_src = src_full[canon].clone()
    can_dst = dst_full[canon].clone()
    n_canon = can_src.size(0)

    if n_canon < 2:
        return edge_index

    n_use = min(2 * n_swaps, n_canon - (n_canon % 2))
    if n_use < 2:
        return edge_index

    perm = torch.randperm(n_canon, device=device)[:n_use]
    i_idx = perm[0::2]
    j_idx = perm[1::2]

    # swap destinations between paired edges
    new_dst_i = can_dst[j_idx].clone()
    new_dst_j = can_dst[i_idx].clone()
    can_dst[i_idx] = new_dst_i
    can_dst[j_idx] = new_dst_j

    # restore canonical form after the swap may have reversed it
    swapped = can_src > can_dst
    final_src = torch.where(swapped, can_dst, can_src)
    final_dst = torch.where(swapped, can_src, can_dst)

    # drop self-loops the swap may have created
    mask = final_src != final_dst
    final_src = final_src[mask]
    final_dst = final_dst[mask]

    # rebuild symmetric edge_index (both directions, as pyg expects)
    new_edge_index = torch.stack([
        torch.cat([final_src, final_dst]),
        torch.cat([final_dst, final_src])
    ])

    return new_edge_index


if __name__ == "__main__":
    import doctest
    doctest.testmod()
