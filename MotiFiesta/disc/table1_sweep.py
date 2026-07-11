"""extended table1 sweep: motif size (k5/k10/k20), multi-motif (m3/m5), sparse.

uses existing trained models evaluated zero-shot on new dataset configurations.
run from repo root:
    python scripts/table1_sweep.py
"""
import json
import statistics

import torch

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.training.loading import get_loader
from MotiFiesta.disc.disc_model import MotiFiestaDisc
from MotiFiesta.disc.disc_decode import _embed_all, _decode_from_cache, _eval

DISTORTIONS = ['0.00']
HASH_DIMS   = [8, 16, 32]
TOP_K       = 3
N_RUNS      = 3
N_GRAPHS    = 200

# (section_label, motif_type, model_name, k, n_planted, sparse)
# n_planted = number of planted motifs per graph (for _eval's n_motifs arg)
EXPERIMENTS = [
    # size sweep — random motif type, parent_size=motif_size*2 (matches paper)
    ('size',   'random',  'random_walk',   5,  1, False),
    ('size',   'random',  'random_walk',  10,  1, False),
    ('size',   'random',  'random_walk',  20,  1, False),
    # multi-motif — random_walk model, motif_size=6 matching original build script
    ('multi',  'multi',   'random_walk',   6,  3, False),
    ('multi',  'multi',   'random_walk',   6,  5, False),
    # sparse — three background density levels
    ('sparse', 'barbell', 'barbell_walk', 10,  1, 0.30),
    ('sparse', 'barbell', 'barbell_walk', 10,  1, 0.50),
    ('sparse', 'barbell', 'barbell_walk', 10,  1, 1.00),
]


def data_path(motif, d, k, n_planted, sparse):
    d_tag = f'-d{d}' if d != '0.00' else ''
    if n_planted > 1:
        return f'data/synth-multi-m{n_planted}-k{k}{d_tag}'
    if isinstance(sparse, float):
        return f'data/synth-{motif}-k{k}-s{sparse:.2f}{d_tag}'
    return f'data/synth-{motif}-k{k}{d_tag}'


def load_disc_model(name, device):
    ckpt = torch.load(f'models/{name}/{name}_best.pth', map_location='cpu', weights_only=False)
    with open(f'models/{name}/hparams.json') as f:
        hp = json.load(f)['model']
    wl_raw    = hp.get('walk_lens', hp.get('walk_len', 8))
    walk_lens = [int(x) for x in wl_raw.split(',')] if isinstance(wl_raw, str) else (
                 wl_raw if isinstance(wl_raw, list) else [wl_raw])
    model = MotiFiestaDisc(
        n_features = hp['n_features'],
        hidden_dim = hp.get('hidden_dim', 32),
        gin_layers = hp.get('gin_layers', 2),
        rwse_steps = hp.get('rwse_steps', 8),
        walk_lens  = walk_lens,
        n_walks    = hp.get('n_walks', 4),
        wl_hops    = hp.get('wl_hops', 1),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


def eval_one(model, dataset, device, n_planted):
    """like eval_config but passes n_planted to _eval."""
    cache = _embed_all(model, dataset, device, N_GRAPHS)
    levels = list(range(len(model.walk_lens)))
    best_mean, best_std = 0.0, 0.0
    for hash_dim in HASH_DIMS:
        for level in levels:
            scores = []
            for _ in range(N_RUNS):
                dg = _decode_from_cache(model, cache, hash_dim, level)
                scores.append(_eval(dg, n_motifs=n_planted, top_k=TOP_K))
            m = statistics.mean(scores)
            s = statistics.stdev(scores) if len(scores) > 1 else 0.0
            if m > best_mean:
                best_mean, best_std = m, s
    return best_mean, best_std


import sys
sys.stdout.reconfigure(line_buffering=True)

device = get_device()
model_cache = {}

# group by section for readable output
sections = {}
for exp in EXPERIMENTS:
    sections.setdefault(exp[0], []).append(exp)

for section, exps in sections.items():
    print(f'\n=== {section} ===')
    col_w = 20
    header = f"{'config':<22}" + ''.join(f"{'ε='+d:>{col_w}}" for d in DISTORTIONS)
    print(header)
    print('-' * len(header))

    for _, motif, model_name, k, n_planted, sparse in exps:
        if model_name not in model_cache:
            print(f'  loading {model_name}...', flush=True)
            model_cache[model_name] = load_disc_model(model_name, device)
        model = model_cache[model_name]

        sparse_tag = f'-s{sparse:.2f}' if isinstance(sparse, float) else ''
        tag = f'{motif}-k{k}{sparse_tag}' + (f'-m{n_planted}' if n_planted > 1 else '')
        row = []
        for d in DISTORTIONS:
            path = data_path(motif, d, k, n_planted, sparse)
            try:
                dataset = get_loader(root=path, name='synth_pairs')
                mean, std = eval_one(model, dataset, device, n_planted)
                row.append(f'{mean:.2f}±{std:.2f}')
            except Exception as e:
                row.append(f'ERR: {str(e)[:10]}')
            print(f'  {tag} d={d}: {row[-1]}', flush=True)

        print(f'{tag:<22}' + ''.join(f'{s:>{col_w}}' for s in row))
