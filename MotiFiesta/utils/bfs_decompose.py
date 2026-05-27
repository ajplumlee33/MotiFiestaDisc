"""
graph decomposition utilities for large-graph MotiFiesta adaptation.

two strategies are provided:
  - BFSDecomposedDataset  / bfs_decompose  : BFS-seeded fixed-size subgraphs
  - LouvainDecomposedDataset / louvain_decompose : community-detection subgraphs
    that keep densely connected groups (e.g. cliques) intact, with a max_size
    ceiling and BFS-based splitting for any over-large community.

both mirror SyntheticMotifs: process() stores {'pos', 'neg'} pairs as individual
files; get() loads and returns the dict; DataLoader collates as usual.

usage:
    from MotiFiesta.utils.bfs_decompose import (
        bfs_decompose, BFSDecomposedDataset,
        louvain_decompose, LouvainDecomposedDataset,
    )
"""
import os
from collections import deque

import networkx as nx
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import subgraph as pyg_subgraph
from MotiFiesta.utils.sys_txt import rewire


def bfs_decompose(data, subgraph_size=20, halo_hops=1, max_degree=None):
    """
    decompose a large pyg graph into small connected subgraphs via BFS.

    seeds from highest-degree nodes first so motif nodes (high local density)
    anchor subgraphs early. halo expansion adds one hop beyond the core so
    motifs straddling a subgraph boundary still appear complete in at least
    one subgraph. halo nodes are reusable across adjacent subgraphs.

    args:
        data: pyg Data with x and edge_index. is_motif / motif_id preserved
              if present.
        subgraph_size: target core node count per subgraph (before halo).
        halo_hops: boundary expansion depth (1 is sufficient in practice).

    returns:
        list of pyg Data objects, one per subgraph.
    """
    if max_degree is None:
        max_degree = subgraph_size * 2

    num_nodes = data.num_nodes
    has_motif_attrs = (
        hasattr(data, 'is_motif') and data.is_motif is not None
        and hasattr(data, 'motif_id') and data.motif_id is not None
    )

    # build plain adjacency list — faster for BFS than nx on large graphs
    adj = [[] for _ in range(num_nodes)]
    src = data.edge_index[0].tolist()
    dst = data.edge_index[1].tolist()
    for u, v in zip(src, dst):
        adj[u].append(v)

    # seed by degree descending — motif nodes have higher degree
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    degrees.scatter_add_(
        0, data.edge_index[0],
        torch.ones(data.edge_index.size(1), dtype=torch.long)
    )
    seed_order = degrees.argsort(descending=True).tolist()

    visited_core = [False] * num_nodes
    subgraph_list = []

    for seed in seed_order:
        if visited_core[seed]:
            continue

        # BFS: collect core nodes up to subgraph_size, skipping already-visited nodes
        core = []
        in_queue = {seed}
        q = deque([seed])
        while q and len(core) < subgraph_size:
            node = q.popleft()
            if visited_core[node]:
                continue
            core.append(node)
            for nb in adj[node]:
                if nb not in in_queue:
                    in_queue.add(nb)
                    q.append(nb)

        # halo: one or more hops beyond the core boundary
        halo = set()
        frontier = set(core)
        for _ in range(halo_hops):
            next_frontier = set()
            for node in frontier:
                for nb in adj[node]:
                    if nb not in in_queue and nb not in halo:
                        halo.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier

        node_idx = torch.tensor(sorted(set(core) | halo), dtype=torch.long)

        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[node_idx] = True
        edge_index_sub, _ = pyg_subgraph(
            node_mask, data.edge_index, relabel_nodes=True, num_nodes=num_nodes
        )

        if edge_index_sub.size(1) == 0:
            for n in core:
                visited_core[n] = True
            continue

        sub = Data(edge_index=edge_index_sub, num_nodes=len(node_idx), n_id=node_idx)
        if has_motif_attrs:
            sub.is_motif = data.is_motif[node_idx]
            sub.motif_id = data.motif_id[node_idx]

        # compute degree features within this subgraph so features reflect
        # local structure, not full-graph degree
        sub = T.OneHotDegree(max_degree)(sub)

        subgraph_list.append(sub)

        for n in core:
            visited_core[n] = True

    return subgraph_list


class BFSDecomposedDataset(Dataset):
    """
    pyg Dataset of BFS-decomposed subgraph pairs.

    mirrors SyntheticMotifs: process() decomposes the large graph, generates
    a degree-preserving rewired negative for each subgraph, and saves each
    {'pos': Data, 'neg': Data} pair as a separate file. get() loads from disk.

    first call with source_data triggers decomposition and caches to disk.
    subsequent calls with the same root load from cache.

    args:
        root: directory for processed cache.
        source_data: pyg Data object of the large graph (required first run).
        subgraph_size: target core node count per subgraph.
        halo_hops: boundary halo depth (default 1).
    """

    def __init__(self, root, source_data=None, subgraph_size=20,
                 halo_hops=1, max_degree=None, transform=None, pre_transform=None):
        self.subgraph_size = subgraph_size
        self.halo_hops = halo_hops
        self.max_degree = max_degree  # None → subgraph_size * 2 in bfs_decompose
        self._source_data = source_data
        super().__init__(root, transform, pre_transform)
        # preload all subgraphs into RAM to avoid random-access disk I/O during training
        self._cache = [
            torch.load(
                os.path.join(self.processed_dir, f'data_{i}.pt'),
                weights_only=False,
            )
            for i in range(self.len())
        ]
        # from_networkx doesn't store num_nodes in _mapping; pyg collator calls
        # store['num_nodes'] directly and crashes. force explicit assignment.
        for pair in self._cache:
            pair['neg'].num_nodes = pair['pos'].num_nodes

    @property
    def num_features(self):
        return torch.load(
            os.path.join(self.processed_dir, 'data_0.pt'),
            weights_only=False,
        )['pos'].num_features

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # checked by pyg to decide whether process() needs to run
        return ['data_0.pt']

    def download(self):
        pass

    def len(self):
        return len([
            f for f in os.listdir(self.processed_dir)
            if f.startswith('data_') and f.endswith('.pt')
        ])

    def get(self, idx):
        return self._cache[idx]

    def process(self):
        if self._source_data is None:
            raise ValueError(
                "source_data required when no processed cache exists. "
                "pass the large-graph pyg Data object as source_data."
            )
        subgraph_list = bfs_decompose(
            self._source_data,
            subgraph_size=self.subgraph_size,
            halo_hops=self.halo_hops,
            max_degree=self.max_degree,
        )
        print(f"decomposed into {len(subgraph_list)} subgraphs "
              f"(core={self.subgraph_size}, halo={self.halo_hops})")

        for i, pos in enumerate(subgraph_list):
            n_edges = pos.edge_index.size(1) // 2
            if n_edges < 2:
                neg = Data(x=pos.x, edge_index=pos.edge_index)
            else:
                n_iter = max(100, min(n_edges * 2, 5000))
                neg = rewire(pos, n_iter=n_iter)
            torch.save({'pos': pos, 'neg': neg},
                       os.path.join(self.processed_dir, f'data_{i}.pt'))

        print(f"saved {len(subgraph_list)} pos/neg pairs to {self.processed_dir}")


# ---------------------------------------------------------------------------
# Louvain-based decomposition
# ---------------------------------------------------------------------------

def _bfs_split(nodes, adj, max_size):
    """split a set of nodes into connected chunks of ≤ max_size via BFS."""
    node_set = set(nodes)
    visited = set()
    groups = []
    for seed in nodes:
        if seed in visited:
            continue
        group = []
        q = deque([seed])
        in_q = {seed}
        while q and len(group) < max_size:
            node = q.popleft()
            if node in visited:
                continue
            group.append(node)
            visited.add(node)
            for nb in adj[node]:
                if nb in node_set and nb not in in_q:
                    in_q.add(nb)
                    q.append(nb)
        groups.append(group)
    return groups


def louvain_decompose(data, max_size=30, min_size=3, resolution=1.0,
                      seed=42, max_degree=None):
    """
    decompose a large pyg graph into subgraphs via Louvain community detection.

    densely connected groups (e.g. cliques) stay intact as a single community.
    communities ≤ max_size are kept as-is; larger ones are split via BFS.
    communities < min_size are discarded.

    args:
        data: pyg Data with edge_index. is_motif / motif_id preserved if present.
        max_size: size ceiling — communities larger than this are BFS-split.
        min_size: communities smaller than this are skipped.
        resolution: Louvain resolution (higher → smaller, more communities).
        seed: RNG seed for reproducible Louvain runs.

    returns:
        list of pyg Data objects, one per community (or BFS chunk).
    """
    if max_degree is None:
        max_degree = max_size * 2

    num_nodes = data.num_nodes
    has_motif_attrs = (
        hasattr(data, 'is_motif') and data.is_motif is not None
        and hasattr(data, 'motif_id') and data.motif_id is not None
    )

    src = data.edge_index[0].tolist()
    dst = data.edge_index[1].tolist()

    # build networkx graph for Louvain (undirected, unweighted)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(src, dst))

    communities = nx.community.louvain_communities(G, resolution=resolution, seed=seed)

    # adjacency list for BFS splitting of oversized communities
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        adj[u].append(v)

    subgraph_list = []
    for community in communities:
        nodes = list(community)

        if len(nodes) < min_size:
            continue

        node_groups = (
            [nodes] if len(nodes) <= max_size
            else _bfs_split(nodes, adj, max_size)
        )

        for node_group in node_groups:
            if len(node_group) < min_size:
                continue

            node_idx = torch.tensor(sorted(node_group), dtype=torch.long)
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[node_idx] = True

            edge_index_sub, _ = pyg_subgraph(
                node_mask, data.edge_index, relabel_nodes=True, num_nodes=num_nodes
            )
            if edge_index_sub.size(1) == 0:
                continue

            sub = Data(edge_index=edge_index_sub, num_nodes=len(node_idx), n_id=node_idx)
            if has_motif_attrs:
                sub.is_motif = data.is_motif[node_idx]
                sub.motif_id = data.motif_id[node_idx]

            sub = T.OneHotDegree(max_degree)(sub)
            subgraph_list.append(sub)

    return subgraph_list


class LouvainDecomposedDataset(Dataset):
    """
    pyg Dataset of Louvain-decomposed subgraph pairs.

    same interface as BFSDecomposedDataset; swap in by pointing build scripts
    at this class. communities keep densely-connected groups intact so motifs
    (e.g. K-cliques) are not severed across subgraph boundaries.

    args:
        root: directory for processed cache.
        source_data: pyg Data of the large graph (required on first run).
        max_size: community size ceiling (oversized communities are BFS-split).
        min_size: communities smaller than this are skipped.
        resolution: Louvain resolution parameter.
        seed: RNG seed for Louvain.
    """

    def __init__(self, root, source_data=None, max_size=30, min_size=3,
                 resolution=1.0, seed=42, max_degree=None,
                 transform=None, pre_transform=None):
        self.max_size = max_size
        self.min_size = min_size
        self.resolution = resolution
        self.seed = seed
        self.max_degree = max_degree
        self._source_data = source_data
        super().__init__(root, transform, pre_transform)
        self._cache = [
            torch.load(
                os.path.join(self.processed_dir, f'data_{i}.pt'),
                weights_only=False,
            )
            for i in range(self.len())
        ]
        for pair in self._cache:
            pair['neg'].num_nodes = pair['pos'].num_nodes

    @property
    def num_features(self):
        return torch.load(
            os.path.join(self.processed_dir, 'data_0.pt'),
            weights_only=False,
        )['pos'].num_features

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_0.pt']

    def download(self):
        pass

    def len(self):
        return len([
            f for f in os.listdir(self.processed_dir)
            if f.startswith('data_') and f.endswith('.pt')
        ])

    def get(self, idx):
        return self._cache[idx]

    def process(self):
        if self._source_data is None:
            raise ValueError(
                "source_data required when no processed cache exists."
            )
        subgraph_list = louvain_decompose(
            self._source_data,
            max_size=self.max_size,
            min_size=self.min_size,
            resolution=self.resolution,
            seed=self.seed,
            max_degree=self.max_degree,
        )
        print(f"decomposed into {len(subgraph_list)} subgraphs "
              f"(max_size={self.max_size}, resolution={self.resolution})")

        for i, pos in enumerate(subgraph_list):
            n_edges = pos.edge_index.size(1) // 2
            if n_edges < 2:
                neg = Data(x=pos.x, edge_index=pos.edge_index, num_nodes=pos.num_nodes)
            else:
                n_iter = max(100, min(n_edges * 2, 5000))
                neg = rewire(pos, n_iter=n_iter)
                neg.num_nodes = pos.num_nodes
            torch.save({'pos': pos, 'neg': neg},
                       os.path.join(self.processed_dir, f'data_{i}.pt'))

        print(f"saved {len(subgraph_list)} pos/neg pairs to {self.processed_dir}")
