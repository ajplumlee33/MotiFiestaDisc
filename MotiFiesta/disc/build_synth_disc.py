"""build synthetic datasets for disc model ablation sweep.

generates size (k5/k20), multi-motif (m3/m5), and sparse datasets.
distortion robustness is covered by existing k10 datasets.
run from repo root:
    python scripts/build_synth_disc.py
"""
import os
import shutil

from MotiFiesta.utils.synthetic import SyntheticMotifs


def build(name, **kwargs):
    proc = f'data/{name}/processed'
    if os.path.exists(proc):
        print(f'  clearing {proc}')
        shutil.rmtree(proc)
    print(f'  building {name}')
    SyntheticMotifs(root='data', name=name, **kwargs)


print('=== size ablation (random motif, parent_size=motif_size*2) ===')
# matches build_data_motifiesta size section
for motif_size in [5, 20]:
    build(f'synth-random-k{motif_size}',
          motif_size=motif_size, motif_type='random',
          parent_size=motif_size * 2, distort_p=0.0)

print('=== multi-motif m3/m5 ===')
# matches build_data_motifiesta multi-motif section: motif_size=6, parent_size=6*n*2
for n in [3, 5]:
    build(f'synth-multi-m{n}-k6',
          motif_size=6, motif_type=None, n_motifs=n,
          parent_size=6 * n * 2, distort_p=0.0)

print('=== sparse (barbell, parent_e_prob = background density) ===')
# 0.30/0.50/1.00 all exceed ER connectivity threshold for n=20 (~0.15)
for sparsity in [0.30, 0.50, 1.00]:
    build(f'synth-barbell-k10-s{sparsity:.2f}',
          motif_size=10, motif_type='barbell', parent_size=20,
          parent_e_prob=sparsity, distort_p=0.0)

print('done. 7 datasets total.')
