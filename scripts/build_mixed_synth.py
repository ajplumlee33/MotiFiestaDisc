"""
build synth-mixed-k10: 500 barbell + 500 star graphs.

  barbell motif nodes → motif_id=1
  star    motif nodes → motif_id=2
  background         → motif_id=0

shuffled so the two types are interleaved during training.

run from repo root:
    python scripts/build_mixed_synth.py
"""
import os
import random
import torch

SRC_BARBELL = 'data/synth-barbell-k10/processed'
SRC_STAR    = 'data/synth-star-k10/processed'
DEST        = 'data/synth-mixed-k10/processed'
N_EACH      = 500

os.makedirs(DEST, exist_ok=True)

pairs = []

for i in range(N_EACH):
    d = torch.load(os.path.join(SRC_BARBELL, f'data_{i}.pt'), map_location='cpu',
                   weights_only=False)
    # barbell motif_id=1 already; no remap needed
    pairs.append(d)

for i in range(N_EACH):
    d = torch.load(os.path.join(SRC_STAR, f'data_{i}.pt'), map_location='cpu',
                   weights_only=False)
    # remap star motif nodes from 1 → 2
    d['pos'].motif_id = d['pos'].motif_id.clone()
    d['pos'].motif_id[d['pos'].motif_id == 1] = 2
    pairs.append(d)

random.seed(42)
random.shuffle(pairs)

for i, pair in enumerate(pairs):
    torch.save(pair, os.path.join(DEST, f'data_{i}.pt'))
    print(f'{i+1}/{len(pairs)}', end='\r')

print(f'\ndone: {len(pairs)} pairs saved to {DEST}')
d0 = pairs[0]
print(f"sample — nodes: {d0['pos'].num_nodes}, motif_ids present: {d0['pos'].motif_id.unique().tolist()}")
