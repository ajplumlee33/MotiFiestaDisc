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
    parser.add_argument('--name',         default='louvain_clique_k10_scatter')
    parser.add_argument('--data',         default='data/louvain_decomp-clique-p0.05-n500-k10-d0.00')
    parser.add_argument('--eval-data',    default=None,
                        help='held-out dataset root for evaluation (avoids in-sample eval)')
    parser.add_argument('--eval-seed',    type=int, default=None,
                        help='seed for held-out eval graph; requires --eval-data')
    parser.add_argument('--motif-type',   default='clique')
    parser.add_argument('--motif-size',   type=int, default=10)
    parser.add_argument('--n-motifs',     type=int, default=20)
    parser.add_argument('--parent-size',  type=int, default=1000)
    parser.add_argument('--parent-e-prob',type=float, default=0.01)
    parser.add_argument('--level',        type=int,   default=3)
    parser.add_argument('--hash-dim',     type=int,   default=8)
    parser.add_argument('--top-k',        type=int,   default=1)
    parser.add_argument('--top-frac',       type=float, default=None,
                        help='fraction of nodes for score-only eval; omit to use Otsu threshold (no oracle)')
    parser.add_argument('--subgraph-scale', action='store_true',
                        help='decode at training batch scale via loader_whole (fixes train/decode mismatch)')
    parser.add_argument('--multilevel', action='store_true',
                        help='combine embeddings from all pooling levels into one hash table')
    parser.add_argument('--n-runs',         type=int,   default=5)
    parser.add_argument('--dataset',        default=None,
                        help="dataset name passed to get_loader (e.g. synth_pairs, louvain_decomp). "
                             "inferred from --data path if omitted.")
    args = parser.parse_args()

    MODEL_ID     = args.name
    DATASET_ROOT = args.data
    EVAL_ROOT    = args.eval_data
    LEVEL        = args.level
    HASH_DIM     = args.hash_dim
    TOP_K        = args.top_k
    N_RUNS       = args.n_runs

    eval_kwargs = {}
    if EVAL_ROOT is not None:
        eval_kwargs = dict(
            motif_type=args.motif_type,
            motif_size=args.motif_size,
            n_motifs=args.n_motifs,
            parent_size=args.parent_size,
            parent_e_prob=args.parent_e_prob,
            seed=args.eval_seed if args.eval_seed is not None else 1,
        )

    decoder = HashDecoder(
        model_id=MODEL_ID,
        dataset_id=DATASET_ROOT,
        eval_dataset_id=EVAL_ROOT,
        eval_kwargs=eval_kwargs,
        hash_dim=HASH_DIM,
        level=LEVEL,
        dataset_name=args.dataset,
    )

    decoder.edge_score_stats(n_graphs=10)

    if args.multilevel:
        decode_fn = decoder.decode_multilevel
    elif args.subgraph_scale:
        decode_fn = decoder.decode_subgraph_scale
    else:
        decode_fn = decoder.decode

    import statistics

    def _run_decode(fn, label):
        sigma_scores, count_scores = [], []
        for _ in range(N_RUNS):
            dg = fn()
            sigma_scores.append(decoder.eval(dg, n_motifs=1, top_k=TOP_K, rank_by='sigma'))
            count_scores.append(decoder.eval(dg, n_motifs=1, top_k=TOP_K, rank_by='count'))
        print(f"\nM-Jaccard over {N_RUNS} runs ({label}, hash_dim={HASH_DIM}):")
        print(f"  sigma: mean={statistics.mean(sigma_scores):.4f}  std={statistics.stdev(sigma_scores):.4f}  runs={[f'{s:.4f}' for s in sigma_scores]}")
        print(f"  count: mean={statistics.mean(count_scores):.4f}  std={statistics.stdev(count_scores):.4f}  runs={[f'{s:.4f}' for s in count_scores]}")
        return dg

    if args.multilevel:
        decoded_graphs = _run_decode(decoder.decode_multilevel, 'multilevel')
        # also report single-level for direct paper comparison
        _run_decode(decoder.decode, f'level={LEVEL}')
    elif args.subgraph_scale:
        decoded_graphs = _run_decode(decoder.decode_subgraph_scale, f'subgraph_scale level={LEVEL}')
    else:
        decoded_graphs = _run_decode(decode_fn, f'level={LEVEL}')

    mot_sigma = decoder.motif_sigma(decoded_graphs)
    print(f"motif sigma by class: {mot_sigma}")

    sigma_thresh = decoder.eval_sigma_threshold(top_frac=args.top_frac)
    print(f"\nscore-only eval (top_frac={args.top_frac}): jaccard={sigma_thresh:.4f}  (no LSH variance)")


if __name__ == "__main__":
    main()
