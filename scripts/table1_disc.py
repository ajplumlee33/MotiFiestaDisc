#!/usr/bin/env python3
"""table1 sweep for MotiFiestaDisc models.

trains on d=0.00, evaluates on d=0.00/0.01/0.02/0.05.
run from repo root:
    python scripts/table1_disc.py
"""
import json
import statistics

import torch

from MotiFiesta.utils.learning_utils import load_model, get_device
from MotiFiesta.training.loading import get_loader
from MotiFiesta.training.disc_model import MotiFiestaDisc
from MotiFiesta.training.disc_decode import eval_config, _embed_all, _decode_from_cache, _eval

MOTIFS = [
    ('barbell', 'barbell_walk'),
    ('clique',  'clique_walk'),
    ('random',  'random_walk'),
    ('star',    'star_walk'),
]

DISTORTIONS = ['0.00', '0.01', '0.02', '0.05']

HASH_DIMS = [8, 16, 32]
LEVELS    = None  # set per-model from walk_lens length
TOP_K     = 3
N_RUNS    = 3
N_GRAPHS  = 200


def load_disc_model(name, device):
    ckpt = torch.load(f'models/{name}/{name}_best.pth', map_location='cpu', weights_only=False)
    msd = ckpt['model_state_dict']
    with open(f'models/{name}/hparams.json') as f:
        hparams = json.load(f)
    n_features    = hparams['model']['n_features']
    rwse_steps    = hparams['model'].get('rwse_steps', 8)
    n_walks       = hparams['model'].get('n_walks', 4)
    wl_raw        = hparams['model'].get('walk_lens', hparams['model'].get('walk_len', 8))
    walk_lens     = [int(x) for x in wl_raw.split(',')] if isinstance(wl_raw, str) else (
        wl_raw if isinstance(wl_raw, list) else [wl_raw])
    wl_hops       = hparams['model'].get('wl_hops', 1)
    hidden_dim    = hparams['model'].get('hidden_dim', 32)
    gin_layers    = hparams['model'].get('gin_layers', 2)
    pair_sampling = hparams['model'].get('pair_sampling', False)
    model = MotiFiestaDisc(n_features=n_features,
                           hidden_dim=hidden_dim, gin_layers=gin_layers,
                           rwse_steps=rwse_steps, walk_lens=walk_lens, n_walks=n_walks,
                           wl_hops=wl_hops, pair_sampling=pair_sampling)
    model.load_state_dict(msd)
    model.to(device)
    model.eval()
    return model


def data_path(motif, d):
    d_tag = f'-d{d}' if d != '0.00' else ''
    return f'data/synth-{motif}-k10{d_tag}'


device = get_device()

# header
col_w = 18
header = f"{'motif':<10}" + ''.join(f"{'ε='+d:>{col_w}}" for d in DISTORTIONS)
print(header)
print('-' * len(header))

n_total = len(MOTIFS) * len(DISTORTIONS) * len(HASH_DIMS)
n_done = 0

for motif, model_name in MOTIFS:
    print(f'\nloading {model_name}...', flush=True)
    model = load_disc_model(model_name, device)
    levels = list(range(len(model.walk_lens)))
    row_scores = []

    for d in DISTORTIONS:
        path = data_path(motif, d)
        dataset = get_loader(root=path, name='synth_pairs')

        print(f'  embedding {N_GRAPHS} graphs for {motif} d={d}...', end=' ', flush=True)
        cache = _embed_all(model, dataset, device, N_GRAPHS)
        print('done', flush=True)

        best_mean, best_std = 0.0, 0.0
        best_hash_dim, best_level = None, None
        for hash_dim in HASH_DIMS:
            for level in levels:
                print(f'  [{n_done+1}/{n_total}] {motif} d={d} hash_dim={hash_dim} level={level}...', end=' ', flush=True)
                mean, std = eval_config(model, dataset, device, hash_dim, level,
                                        TOP_K, N_RUNS, N_GRAPHS, cache=cache)
                print(f'{mean:.3f}', flush=True)
                n_done += 1
                if mean > best_mean:
                    best_mean, best_std = mean, std
                    best_hash_dim, best_level = hash_dim, level

        print(f'  dummy at hash_dim={best_hash_dim} level={best_level}...', end=' ', flush=True)
        dummy_scores = []
        for _ in range(N_RUNS):
            dg = _decode_from_cache(model, cache, best_hash_dim, best_level, dummy=True)
            dummy_scores.append(_eval(dg, n_motifs=1, top_k=TOP_K))
        dm = statistics.mean(dummy_scores)
        ds = statistics.stdev(dummy_scores) if len(dummy_scores) > 1 else 0.0
        print(f'{dm:.3f}', flush=True)

        cell = f'{best_mean:.2f}±{best_std:.2f} ({dm:.2f}±{ds:.2f})'
        row_scores.append(cell)
        print(f'  → {motif} d={d}: {cell}', flush=True)

    print(flush=True)
    print(f'{motif:<10}' + ''.join(f'{s:>{col_w}}' for s in row_scores))

print()

import sys; sys.exit(0)
