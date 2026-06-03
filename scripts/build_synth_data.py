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

MOTIF_TYPE   = 'clique'    # 'clique', 'star', 'barbell', 'wheel', 'lollipop', 'random'
MOTIF_SIZE   = 10
PARENT_SIZE  = 20          # background nodes; total graph ≈ motif_size + parent_size (~2x motif size, paper appendix A.1)
PARENT_EPROB = 0.1         # paper appendix A.1
MAX_DEGREE   = 61
DISTORT_P    = -1          # no distortion
N_GRAPHS     = 1000
USE_EIGEN_PE = False       # append laplacian eigenvectors to node features
N_EIGEN      = 8           # number of eigenvectors (ignored if USE_EIGEN_PE=False)

_eig_suffix = f'-eig{N_EIGEN}' if USE_EIGEN_PE else ''
DEST_ROOT = f'data/synth-{MOTIF_TYPE}-k{MOTIF_SIZE}{_eig_suffix}'


def laplacian_pe(data, k):
    """append k laplacian eigenvectors to data.x in-place."""
    n = data.num_nodes
    k_actual = min(k, n - 1)
    if k_actual < 1:
        data.x = torch.cat([data.x, torch.zeros(n, k)], dim=1)
        return data

    edge_index = data.edge_index
    adj = torch.zeros(n, n)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e6)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L = torch.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

    _, evecs = torch.linalg.eigh(L)
    # skip trivial eigenvector (index 0), take next k_actual
    pe = evecs[:, 1:k_actual + 1]

    # sign normalization: largest-magnitude element positive per eigenvector
    signs = pe[pe.abs().argmax(dim=0), torch.arange(k_actual)].sign()
    signs[signs == 0] = 1.0
    pe = pe * signs

    if k_actual < k:
        pe = torch.cat([pe, torch.zeros(n, k - k_actual)], dim=1)

    data.x = torch.cat([data.x, pe.float()], dim=1)
    return data


processed_dir = os.path.join(DEST_ROOT, 'processed')
os.makedirs(processed_dir, exist_ok=True)

print(f"generating {N_GRAPHS} {MOTIF_TYPE}-k{MOTIF_SIZE} graphs "
      f"{'(+eigen PE) ' if USE_EIGEN_PE else ''}...")
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
    neg = triplet['neg']   # background-only (no motif) — original paper setup
    pos.num_nodes = pos.x.size(0)
    neg.num_nodes = neg.x.size(0)
    if USE_EIGEN_PE:
        pos = laplacian_pe(pos, N_EIGEN)
        neg = laplacian_pe(neg, N_EIGEN)
    torch.save({'pos': pos, 'neg': neg},
               os.path.join(processed_dir, f'data_{i}.pt'))

print(f"done: {N_GRAPHS} pairs saved to {DEST_ROOT}")
sample_pos = gs[0]['pos']
print(f"sample — nodes: {sample_pos.x.size(0)}, "
      f"features: {sample_pos.x.shape[1] + (N_EIGEN if USE_EIGEN_PE else 0)}, "
      f"motif nodes: {int(sample_pos.is_motif.sum())}")
