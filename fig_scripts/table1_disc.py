"""
sweep hash_dim x k_group x dataset for rand-esu disc, matching motifiesta table1 structure.
k_groups mirror motifiesta edge pooling levels: level2=k(3-4), level3=k(5-8), level4=k(9-16)
"""
import argparse, os, sys, subprocess, itertools, multiprocessing
import pandas as pd
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECODE = os.path.join(REPO, 'MotiFiesta', 'disc', 'decode.py')
DATA   = os.path.join(REPO, 'data')

DEFAULT_DATASETS = ['synth-barbell-k10', 'synth-clique-k10', 'synth-random-k10', 'synth-star-k10']
HASH_DIMS   = [8, 16, 32]
K_GROUPS    = [(3, 4), (5, 8), (9, 16)]   # analog of motifiesta levels 2, 3, 4
K_INDIVIDUAL = [(k, k) for k in range(3, 17)]  # one k at a time for subgraph mode
N_GRAPHS    = 1000
N_RUNS      = 3
EMBED_DIM   = 32
N_SAMPLES   = 200
NUM_WORKERS = multiprocessing.cpu_count()


def run_once(args):
    dataset, hash_dim, k_min, k_max, run, top_k, proj, score_alpha, embed_mode, pred_mode, score_mode, gae_model, clf = args
    name = f"{dataset}_h{hash_dim}_k{k_min}-{k_max}_r{run}"
    cmd = [
        sys.executable, DECODE,
        '--name',        name,
        '--data',        os.path.join(DATA, dataset),
        '--hash-dim',    str(hash_dim),
        '--embed-dim',   str(EMBED_DIM),
        '--k-min',       str(k_min),
        '--k-max',       str(k_max),
        '--n-graphs',    str(N_GRAPHS),
        '--n-samples',   str(N_SAMPLES),
        '--top-k',       str(top_k),
        '--proj',        proj,
        '--embed-mode',  embed_mode,
        '--pred-mode',   pred_mode,
        '--score-mode',  score_mode,
        '--score-alpha', str(score_alpha),
        '--clf',         clf,
        *(['--gae-model', gae_model] if gae_model else []),
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = REPO
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    jaccard = None
    for line in result.stdout.splitlines():
        if line.startswith('jaccard:'):
            jaccard = float(line.split()[1])
    return {'dataset': dataset, 'hash_dim': hash_dim, 'k_min': k_min,
            'k_max': k_max, 'run': run, 'jaccard': jaccard}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-k',       type=int,   default=3)
    ap.add_argument('--proj',        default='lda', choices=['lda', 'pca'])
    ap.add_argument('--embed-mode',  default='wl', choices=['wl', 'wwl', 'gae', 'spectral', 'tree', 'canonical'])
    ap.add_argument('--clf',         default='auto', choices=['auto', 'lda', 'tree'])
    ap.add_argument('--gae-model',   default=None, metavar='PATH')
    ap.add_argument('--pred-mode',    default='bucket', choices=['bucket', 'subgraph'])
    ap.add_argument('--score-mode',   default='gstat', choices=['count', 'knn', 'gstat'])
    ap.add_argument('--score-alpha',  type=float, default=1.5)
    ap.add_argument('--k-individual', action='store_true',
                    help='sweep each k value individually instead of grouped ranges')
    ap.add_argument('--datasets',    nargs='+',  default=DEFAULT_DATASETS)
    args = ap.parse_args()

    k_groups = K_INDIVIDUAL if args.k_individual else K_GROUPS
    jobs = [
        (dataset, hash_dim, k_min, k_max, run,
         args.top_k, args.proj, args.score_alpha, args.embed_mode, args.pred_mode, args.score_mode, args.gae_model, args.clf)
        for dataset, hash_dim, (k_min, k_max) in itertools.product(args.datasets, HASH_DIMS, k_groups)
        for run in range(N_RUNS)
    ]

    print(f"{len(jobs)} total runs  workers={NUM_WORKERS}"
          f"  proj={args.proj}  alpha={args.score_alpha}"
          f"  embed={args.embed_mode}  n_samples={N_SAMPLES}  n_graphs={N_GRAPHS}\n")

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = []
        for r in pool.imap_unordered(run_once, jobs):
            j = r['jaccard']
            print(f"  {r['dataset']}  h={r['hash_dim']}  k=[{r['k_min']},{r['k_max']}]"
                  f"  run={r['run']+1}  j={j:.4f}" if j is not None else
                  f"  {r['dataset']}  h={r['hash_dim']}  k=[{r['k_min']},{r['k_max']}]"
                  f"  run={r['run']+1}  FAILED", flush=True)
            results.append(r)

    # aggregate runs
    records = []
    for (dataset, hash_dim, k_min, k_max), grp in pd.DataFrame(results).groupby(
            ['dataset', 'hash_dim', 'k_min', 'k_max']):
        jaccards = grp['jaccard'].dropna().tolist()
        if jaccards:
            records.append({
                'dataset':  dataset,
                'hash_dim': hash_dim,
                'k_min':    k_min,
                'k_max':    k_max,
                'level':    k_min if args.k_individual else K_GROUPS.index((k_min, k_max)) + 2,
                'j_mean':   np.mean(jaccards),
                'j_std':    np.std(jaccards),
            })

    df = pd.DataFrame(records)
    dump = os.path.join(REPO, 'fig_scripts',
                        f'table1_disc_{args.proj}_a{args.score_alpha}_n{N_SAMPLES}_{args.embed_mode}.csv')
    df.to_csv(dump, index=False)
    print(f"\nsaved to {dump}\n")

    best = df.loc[df.groupby('dataset')['j_mean'].idxmax()]
    print(best[['dataset', 'hash_dim', 'level', 'j_mean', 'j_std']].to_string(index=False))


if __name__ == '__main__':
    main()
