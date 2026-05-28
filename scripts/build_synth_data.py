"""
generate a SyntheticMotifs dataset in the pos/neg format expected by
MotiFiesta training and the HashDecoder.

SyntheticMotifs.generate_instances returns triplets:
  planted  = motif embedded in ER background  ← pos
  original = background only
  wired    = rewired version of planted        ← neg

saves {'pos': planted_pyg, 'neg': wired_pyg} to a louvain_decomp-prefixed
root so get_loader routes to LouvainDecomposedDataset.

run from repo root:
    python scripts/build_synth_data.py
"""
import os
import torch
from MotiFiesta.utils.synthetic import generate_instances

MOTIF_TYPE  = 'star'     # 'clique', 'star', 'barbell', 'wheel', 'lollipop'
MOTIF_SIZE  = 10
PARENT_SIZE = 10         # background nodes; total graph ≈ motif_size + parent_size
PARENT_EPROB = 0.1       # paper appendix A.1
MAX_DEGREE  = 25
DISTORT_P   = -1         # no distortion
N_GRAPHS    = 1000

DEST_ROOT = f'data/louvain_decomp-synth-{MOTIF_TYPE}-k{MOTIF_SIZE}'

processed_dir = os.path.join(DEST_ROOT, 'processed')
os.makedirs(processed_dir, exist_ok=True)

print(f"generating {N_GRAPHS} {MOTIF_TYPE}-k{MOTIF_SIZE} graphs ...")
gs = generate_instances(
    n_graphs=N_GRAPHS,
    motif_type=MOTIF_TYPE,
    motif_size=MOTIF_SIZE,
    parent_size=PARENT_SIZE,
    parent_e_prob=PARENT_EPROB,
    max_degree=MAX_DEGREE,
    distort_p=DISTORT_P,
    attributed=False,
)

for i, triplet in enumerate(gs):
    pos = triplet['pos']   # planted = motif embedded in background
    neg = triplet['rand']  # wired   = rewired version, same node count
    pos.num_nodes = pos.x.size(0)
    neg.num_nodes = pos.num_nodes
    torch.save({'pos': pos, 'neg': neg},
               os.path.join(processed_dir, f'data_{i}.pt'))

print(f"done: {N_GRAPHS} pairs saved to {DEST_ROOT}")
sample_pos = gs[0]['pos']
print(f"sample — nodes: {sample_pos.x.size(0)}, "
      f"features: {sample_pos.x.shape[1]}, "
      f"motif nodes: {int(sample_pos.is_motif.sum())}")
