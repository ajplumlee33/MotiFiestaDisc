#!/usr/bin/env python3
"""
decode the trained model and evaluate with node-level permutation M-Jaccard.
averages over N_RUNS to smooth LSH variance.

run from repo root:
    python scripts/run_decode.py
"""
import argparse
from MotiFiesta.training.decode import HashDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',      default='louvain_clique_k10_scatter')
    parser.add_argument('--data',      default='data/louvain_decomp-clique-p0.05-n500-k10-d0.00')
    parser.add_argument('--level',     type=int,   default=1)
    parser.add_argument('--hash-dim',  type=int,   default=8)
    parser.add_argument('--top-k',     type=int,   default=1)
    parser.add_argument('--n-runs',    type=int,   default=5)
    args = parser.parse_args()

    MODEL_ID     = args.name
    DATASET_ROOT = args.data
    LEVEL        = args.level
    HASH_DIM     = args.hash_dim
    TOP_K        = args.top_k
    N_RUNS       = args.n_runs

    decoder = HashDecoder(
        model_id=MODEL_ID,
        dataset_id=DATASET_ROOT,
        hash_dim=HASH_DIM,
        level=LEVEL,
    )

    decoder.edge_score_stats(n_graphs=10)

    sigma_scores, count_scores = [], []
    for _ in range(N_RUNS):
        decoded_graphs = decoder.decode()
        sigma_scores.append(decoder.eval(decoded_graphs, n_motifs=1, top_k=TOP_K, rank_by='sigma'))
        count_scores.append(decoder.eval(decoded_graphs, n_motifs=1, top_k=TOP_K, rank_by='count'))

    import statistics
    print(f"\nM-Jaccard over {N_RUNS} runs (level={LEVEL}, hash_dim={HASH_DIM}):")
    print(f"  sigma: mean={statistics.mean(sigma_scores):.4f}  std={statistics.stdev(sigma_scores):.4f}  runs={[f'{s:.4f}' for s in sigma_scores]}")
    print(f"  count: mean={statistics.mean(count_scores):.4f}  std={statistics.stdev(count_scores):.4f}  runs={[f'{s:.4f}' for s in count_scores]}")

    mot_sigma = decoder.motif_sigma(decoded_graphs)
    print(f"motif sigma by class: {mot_sigma}")


if __name__ == "__main__":
    main()
