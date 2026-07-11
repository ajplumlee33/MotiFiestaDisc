"""generate synth_pairs datasets for a given motif type and distortion level. usage:
    python scripts/gen_synth.py clique
    python scripts/gen_synth.py random 0.01
"""
import os
import sys
import os.path as osp
import torch
from MotiFiesta.utils.synthetic import generate_instances

MOTIF_SIZE   = 10
PARENT_SIZE  = 20
PARENT_EPROB = 0.1
MAX_DEGREE   = 25
N_GRAPHS     = 1000
SEED         = 42

motif_type = sys.argv[1]
distort_p  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

d_tag = f'-d{distort_p:.2f}' if distort_p > 0 else ''
out_dir = osp.join('data', f'synth-{motif_type}-k{MOTIF_SIZE}{d_tag}', 'processed')
os.makedirs(out_dir, exist_ok=True)

print(f"generating {N_GRAPHS} {motif_type} k{MOTIF_SIZE} distort={distort_p} (seed={SEED})...")
instances = generate_instances(
    motif_type=motif_type,
    motif_size=MOTIF_SIZE,
    parent_size=PARENT_SIZE,
    parent_e_prob=PARENT_EPROB,
    max_degree=MAX_DEGREE,
    n_graphs=N_GRAPHS,
    seed=SEED,
    distort_p=distort_p,
    attributed=False,
)

for i, triplet in enumerate(instances):
    data = {'pos': triplet['pos'], 'neg': triplet['neg']}
    torch.save(data, osp.join(out_dir, f'data_{i}.pt'))

print(f"saved {len(instances)} files to {out_dir}")
sample = instances[0]
print(f"sample: pos nodes={sample['pos'].x.size(0)}, x_dim={sample['pos'].x.size(1)}, neg nodes={sample['neg'].x.size(0)}")
