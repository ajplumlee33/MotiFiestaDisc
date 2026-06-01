#!/usr/bin/env python3
"""
diagnose edge score distributions for barbell motif:
  - bridge edges (between the two K5 halves)
  - k5 internal edges (within motif cliques)
  - background edges

run from repo root:
    python scripts/diag_barbell_scores.py --name synth_barbell_k10_edgectx
"""
import argparse
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

from MotiFiesta.utils.learning_utils import load_model, get_device

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='synth_barbell_k10_edgectx')
parser.add_argument('--data', default='data/louvain_decomp-synth-barbell-k10')
parser.add_argument('--n-graphs', type=int, default=20)
parser.add_argument('--level', type=int, default=0)
args = parser.parse_args()

device = get_device()
model_dict = load_model(args.name)
model = model_dict['model'].to(device)
model.eval()

import os
processed = os.path.join(args.data, 'processed')
files = sorted(os.listdir(processed))[:args.n_graphs]

bridge_scores, k5_scores, bg_scores = [], [], []
bridge_logits, k5_logits, bg_logits = [], [], []

for fname in files:
    d = torch.load(os.path.join(processed, fname), weights_only=False)
    pos = d['pos']
    x = pos.x.to(device)
    ei = pos.edge_index.to(device)
    is_motif = pos.is_motif.bool().to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        xx, pp, ee, batches, merge_info, internals = model(x, ei, batch)

    if args.level >= len(internals):
        print(f"level {args.level} not available, max={len(internals)-1}")
        break

    edge_scores = internals[args.level]['edge_scores']
    edge_logits = internals[args.level]['edge_logits']
    edge_idx    = internals[args.level]['edge_index']

    # classify each edge
    src = edge_idx[0]
    dst = edge_idx[1]

    # identify bridge nodes: motif nodes with degree < k5 clique internal degree
    # k5 internal node degree = 4 (within clique) + possibly bridge connections
    # bridge intermediate nodes (path nodes 5,6): degree 2, both in is_motif
    # clique connector nodes (4,7): degree 5 in motif subgraph
    # inner clique nodes (0-3, 8-11): degree 4 in motif subgraph
    # we identify bridge edges as edges where BOTH endpoints are motif nodes
    # but at least one has low motif-subgraph degree

    motif_nodes = is_motif.nonzero(as_tuple=False).squeeze(-1)
    # motif-subgraph degree: count edges where both endpoints are motif nodes
    motif_src_mask = is_motif[src] & is_motif[dst]
    motif_ei = edge_idx[:, motif_src_mask]
    motif_deg = torch.zeros(x.size(0), dtype=torch.long, device=device)
    motif_deg.scatter_add_(0, motif_ei[0], torch.ones(motif_ei.size(1), dtype=torch.long, device=device))

    # bridge nodes: motif nodes with motif-degree <= 2 (the path nodes + connectors)
    # path intermediate nodes have motif-degree 2; clique connectors have motif-degree 5
    # inner clique nodes have motif-degree 4
    bridge_node = is_motif & (motif_deg <= 2)

    for i in range(edge_scores.size(0)):
        u, v = src[i].item(), dst[i].item()
        s = edge_scores[i].item()
        l = edge_logits[i].item() if edge_logits is not None else float('nan')
        u_motif = is_motif[u].item()
        v_motif = is_motif[v].item()
        u_bridge = bridge_node[u].item()
        v_bridge = bridge_node[v].item()

        if u_motif and v_motif:
            if u_bridge or v_bridge:
                bridge_scores.append(s)
                bridge_logits.append(l)
            else:
                k5_scores.append(s)
                k5_logits.append(l)
        else:
            bg_scores.append(s)
            bg_logits.append(l)

def stats(lst, name):
    if not lst:
        print(f"  {name}: no edges")
        return
    t = torch.tensor(lst)
    print(f"  {name:20s}: n={len(lst):4d}  mean={t.mean():.4f}  std={t.std():.4f}  "
          f"min={t.min():.4f}  max={t.max():.4f}")

print(f"\nedge score distributions at level {args.level} ({args.n_graphs} graphs):")
stats(k5_scores,    'k5 internal')
stats(bridge_scores,'bridge')
stats(bg_scores,    'background')

print(f"\nedge logit distributions at level {args.level}:")
stats(k5_logits,    'k5 internal')
stats(bridge_logits,'bridge')
stats(bg_logits,    'background')
