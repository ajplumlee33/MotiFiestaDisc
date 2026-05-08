#!/usr/bin/env python
"""
Generate single-graph synthetic datasets mirroring the paper Table 1 sweep.
"""
from MotiFiesta.utils.sys_synthetic import SysSyntheticDataset

### OPTIONS


"""
n_motifs: number of motif instances to plant in the source graph
motif_type: motif topology
motif_size: number of nodes in motif
parent_size: number of nodes in parent graph
parent_e_prob: edge probability for parent graph
random_e_prob: edge probability for motif when motif_type='random'
distort_p: probability of swapping edges in motif graph
"""


# Motif type
print(">>> MOTIF TYPE")
for m_type in ['barbell', 'star', 'random', 'clique']:
    for d in [0, .01, .02, .05, .1, .2]:
        SysSyntheticDataset(root=f'data/sys_synth-{m_type}-d{d:.2f}', seed=42, motif_type=m_type, motif_size=10, n_motifs=200, parent_size=4000, parent_e_prob=0.0017, random_e_prob=0.3, distort_p=d)

# Size
print(">>> SIZE")
for motif_size in [5, 10, 20]:
    for d in [0, .01, .02, .05, .1, .2]:
        SysSyntheticDataset(root=f'data/sys_synth-size{motif_size:02d}-d{d:.2f}', seed=42, motif_type='clique', motif_size=motif_size, n_motifs=int(round(0.5 * 4000 / motif_size)), parent_size=4000, parent_e_prob=0.0017, distort_p=d)

# Density
print(">>> DENSITY")
for density in [.10, .20, .33]:
    for d in [0, .01, .02, .05, .1, .2]:
        SysSyntheticDataset(root=f'data/sys_synth-dens{density:.2f}-d{d:.2f}', seed=42, motif_type='clique', motif_size=10, n_motifs=int(round(density * 4000 / 10)), parent_size=4000, parent_e_prob=0.0017, distort_p=d)

# Multi-motif
print(">>> MULTI MOTIF")
for types in [['clique', 'star', 'barbell'], ['clique', 'star', 'barbell', 'wheel', 'random']]:
    for d in [0, .01, .02, .05, .1, .2]:
        SysSyntheticDataset(root=f'data/sys_synth-{len(types)}motifs-d{d:.2f}', seed=42, motif_type=types, motif_size=10, n_motifs=50, parent_size=4000, parent_e_prob=0.0017, random_e_prob=0.3, distort_p=d)
