#!/usr/bin/env python3
"""
evaluate DiscModel with the same sweep as table1.py:
  hash_dim in [8, 16, 32], level in [2, 3, 4], top_k=3, n_runs=3, n_graphs=200
  reports best (hash_dim, level) combination and dummy baseline.

run from repo root:
    python scripts/run_decode.py --name barbell_dp14 --data data/synth-barbell-k10 --dataset synth_pairs
"""
import argparse
import statistics
from MotiFiesta.training.decode import HashDecoder


def eval_config(decoder, hash_dim, level, top_k, n_runs, n_graphs):
    decoder.hash_dim = hash_dim
    decoder.level = level
    scores = []
    for _ in range(n_runs):
        dg = decoder.decode(n_graphs=n_graphs)
        scores.append(decoder.eval(dg, n_motifs=1, top_k=top_k, rank_by='sigma'))
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',     required=True)
    parser.add_argument('--data',     required=True)
    parser.add_argument('--dataset',  default=None)
    parser.add_argument('--n-runs',   type=int, default=3)
    parser.add_argument('--n-graphs', type=int, default=200)
    parser.add_argument('--top-k',    type=int, default=3)
    parser.add_argument('--quick',    action='store_true',
                        help='fast sanity check: single config, n_graphs=30, n_runs=1')
    args = parser.parse_args()

    if args.quick:
        args.n_graphs = 30
        args.n_runs = 1

    decoder = HashDecoder(
        model_id=args.name, dataset_id=args.data,
        hash_dim=8, level=3,
        dataset_name=args.dataset,
    )

    print(f"\n=== {args.name} (table1 sweep: hash_dim∈[8,16,32] × level∈[2,3,4], top_k={args.top_k}, n_runs={args.n_runs}, n_graphs={args.n_graphs}) ===\n")

    if args.quick:
        mean, std = eval_config(decoder, 16, 3, args.top_k, args.n_runs, args.n_graphs)
        decoder.dummy = True
        dg = decoder.decode(n_graphs=args.n_graphs)
        dm = decoder.eval(dg, n_motifs=1, top_k=args.top_k, rank_by='sigma')
        decoder.dummy = False
        print(f"quick  sigma={mean:.4f} ±{std:.4f}  dummy={dm:.4f}")
        return

    best = {'mean': 0.0, 'std': 0.0, 'hash_dim': None, 'level': None}
    results = []

    for hash_dim in [8, 16, 32]:
        for level in [2, 3, 4]:
            mean, std = eval_config(decoder, hash_dim, level, args.top_k, args.n_runs, args.n_graphs)
            results.append((hash_dim, level, mean, std))
            marker = ' *' if mean > best['mean'] else ''
            print(f"  hash_dim={hash_dim}  level={level}:  {mean:.4f} ±{std:.4f}{marker}")
            if mean > best['mean']:
                best = {'mean': mean, 'std': std, 'hash_dim': hash_dim, 'level': level}

    # dummy baseline at best config
    decoder.hash_dim = best['hash_dim']
    decoder.level = best['level']
    dummy_scores = []
    for _ in range(args.n_runs):
        dg = decoder.decode(n_graphs=args.n_graphs)
        dummy_scores.append(decoder.eval(dg, n_motifs=1, top_k=args.top_k, rank_by='sigma'))

    # temporarily enable dummy scoring
    decoder.dummy = True
    dummy_scores = []
    for _ in range(args.n_runs):
        dg = decoder.decode(n_graphs=args.n_graphs)
        dummy_scores.append(decoder.eval(dg, n_motifs=1, top_k=args.top_k, rank_by='sigma'))
    decoder.dummy = False
    dm = statistics.mean(dummy_scores)
    ds = statistics.stdev(dummy_scores) if len(dummy_scores) > 1 else 0.0

    print(f"\nbest:  {best['mean']:.2f} ±{best['std']:.2f}  (hash_dim={best['hash_dim']}, level={best['level']})")
    print(f"dummy: {dm:.2f} ±{ds:.2f}")
    print(f"\nfinal: {best['mean']:.2f} ±{best['std']:.2f}  ({dm:.2f} ±{ds:.2f})")


if __name__ == "__main__":
    main()
