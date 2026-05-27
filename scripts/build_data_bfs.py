"""
generate a large synthetic graph and decompose it into small subgraphs
for MotiFiesta training.

two decomposition strategies are available via DECOMP_METHOD:
  'bfs'     - BFS-seeded fixed-size subgraphs (original approach)
  'louvain' - Louvain community detection; keeps cliques intact, variable size
              up to MAX_SIZE ceiling

source graph: ER(n=SOURCE_SIZE, p=SOURCE_EPROB) background + planted K-clique
motifs. multiple source graphs (different seeds) are aggregated to reach
N_GRAPHS * subgraphs_per_source total training pairs.

key constraint: background density within a subgraph ≈ C(subgraph_size, 2)
× SOURCE_EPROB. SOURCE_EPROB=0.1 gives ~19 background edges per 20-node
subgraph, matching the paper's ER(20, p=0.1) small-graph distribution.
SOURCE_SIZE must stay small enough that p=0.1 doesn't produce a near-complete
graph (~500 is a good ceiling).

run from repo root:
    python scripts/build_data_bfs.py
"""
import os
import random

import networkx as nx
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx

from MotiFiesta.utils.bfs_decompose import bfs_decompose, louvain_decompose

DECOMP_METHOD = 'louvain'   # 'bfs' or 'louvain'

MOTIF_TYPE    = 'clique'
MOTIF_SIZE    = 10
SOURCE_SIZE   = 500    # nodes in each source graph; p=0.1 is feasible at this scale
SOURCE_EPROB  = 0.05   # louvain: lower p keeps bg communities sparser than K10 cliques
N_MOTIFS      = 25     # motifs per source graph: 25×10 = 250 motif nodes = 50% coverage
DISTORT_P     = 0.0
N_GRAPHS      = 40     # source graphs to generate
SEED_BASE     = 42

# BFS params (used when DECOMP_METHOD='bfs')
SUBGRAPH_SIZE = MOTIF_SIZE * 2  # 20 nodes; matches paper's 2×motif_size
HALO_HOPS     = 1

# Louvain params (used when DECOMP_METHOD='louvain')
MAX_SIZE      = 30      # any community ≤ MAX_SIZE is kept whole; larger ones are BFS-split
MIN_SIZE      = 3       # communities smaller than this are discarded
RESOLUTION    = 2.0     # higher → smaller communities; 2.0 gives better K10 purity than 1.5

DEST_ROOT = (
    f'data/{DECOMP_METHOD}_decomp-{MOTIF_TYPE}'
    f'-p{SOURCE_EPROB}-n{SOURCE_SIZE}-k{MOTIF_SIZE}'
    f'-d{DISTORT_P:.2f}'
)


def make_neg_er(pos, source_eprob, seed):
    """fresh ER graph with same node count as pos — no motif, same feature dim."""
    n = pos.num_nodes
    max_degree = pos.num_features - 1  # OneHotDegree(d) → d+1 features
    G = nx.erdos_renyi_graph(n, source_eprob, seed=seed)
    data = from_networkx(G)
    data.num_nodes = n
    if data.edge_index is None or data.edge_index.numel() == 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
    data = T.OneHotDegree(max_degree)(data)
    return data


def make_source_graph(motif_size, source_size, source_eprob, n_motifs, seed):
    """ER background graph with n_motifs K-clique instances planted."""
    rng = random.Random(seed)
    G = nx.erdos_renyi_graph(source_size, source_eprob, seed=seed)
    nx.set_node_attributes(G, 0, 'is_motif')
    nx.set_node_attributes(G, 0, 'motif_id')

    offset = G.number_of_nodes()
    for inst in range(n_motifs):
        motif_nodes = list(range(offset, offset + motif_size))
        for n in motif_nodes:
            G.add_node(n, is_motif=1, motif_id=1)
        for i in motif_nodes:
            for j in motif_nodes:
                if i < j:
                    G.add_edge(i, j)
        # random cross edges between motif instance and background
        for m in motif_nodes:
            for b in range(source_size):
                if rng.random() < source_eprob:
                    G.add_edge(m, b)
        offset += motif_size

    G = nx.convert_node_labels_to_integers(G)
    data = from_networkx(G)
    data.is_motif = data.is_motif.long()
    data.motif_id = data.motif_id.long()
    return data


def decompose(data, seed):
    if DECOMP_METHOD == 'bfs':
        return bfs_decompose(data, subgraph_size=SUBGRAPH_SIZE, halo_hops=HALO_HOPS)
    elif DECOMP_METHOD == 'louvain':
        return louvain_decompose(
            data, max_size=MAX_SIZE, min_size=MIN_SIZE,
            resolution=RESOLUTION, seed=seed,
        )
    else:
        raise ValueError(f"unknown DECOMP_METHOD: {DECOMP_METHOD!r}")


def main():
    processed_dir = os.path.join(DEST_ROOT, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    all_subgraphs = []
    for i in range(N_GRAPHS):
        data = make_source_graph(
            MOTIF_SIZE, SOURCE_SIZE, SOURCE_EPROB, N_MOTIFS,
            seed=SEED_BASE + i,
        )
        subs = decompose(data, seed=SEED_BASE + i)
        all_subgraphs.extend(subs)
        if (i + 1) % 10 == 0:
            print(f"  source graph {i+1}/{N_GRAPHS}: "
                  f"{data.num_nodes} nodes, "
                  f"{int(data.is_motif.sum())} motif nodes, "
                  f"→ {len(subs)} subgraphs (total so far: {len(all_subgraphs)})")

    print(f"\ntotal subgraphs: {len(all_subgraphs)}")
    g = all_subgraphs[0]
    print(f"sample — nodes: {g.num_nodes}, "
          f"edges: {g.edge_index.size(1)//2}, "
          f"features: {g.num_features}, "
          f"motif%: {g.is_motif.float().mean():.1%}")

    motif_subgraphs = [s for s in all_subgraphs if s.is_motif.sum() > 0]
    print(f"motif-containing: {len(motif_subgraphs)}/{len(all_subgraphs)} "
          f"({len(motif_subgraphs)/len(all_subgraphs):.1%})")

    print(f"saving pos/neg pairs to {processed_dir} ...")
    for i, pos in enumerate(motif_subgraphs):
        neg = make_neg_er(pos, SOURCE_EPROB, seed=SEED_BASE + i)
        torch.save({'pos': pos, 'neg': neg},
                   os.path.join(processed_dir, f'data_{i}.pt'))

    print(f"done: {len(motif_subgraphs)} pairs saved to {DEST_ROOT}")


if __name__ == '__main__':
    main()
