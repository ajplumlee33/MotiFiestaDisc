#!/usr/bin/env python3
"""
diagnose freq_loss signal quality for DiscModel:
  - delta_f (rho_pos - rho_neg) distributions by motif vs background class
  - score distributions by class at each level
  - transform output magnitudes across levels
  - optimal s* = exp(-beta*delta_f) / (2*lam) vs actual s

run from repo root:
    python scripts/diag_disc_signal.py --name clique_dp7 --data data/synth-clique-k10
    python scripts/diag_disc_signal.py --name barbell_dp7 --data data/synth-barbell-k10
"""
import argparse
import torch
import torch.nn.functional as F

from MotiFiesta.utils.learning_utils import load_model, get_device
from MotiFiesta.training.loading import get_loader

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--dataset', default='synth_pairs')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--k', type=int, default=30)
parser.add_argument('--n-batches', type=int, default=5)
args = parser.parse_args()

device = get_device()
model_dict = load_model(args.name)
model = model_dict['model'].to(device)
model.eval()

loaders = get_loader(args.data, name=args.dataset, batch_size=50)
loader = loaders['loader_test']


def knn_radius(X, X_ref, k):
    neg_sim = -(X @ X_ref.T)
    k_eff = min(k, X.size(0) - 1, X_ref.size(0))
    knn, _ = neg_sim.topk(k_eff, dim=1, largest=False)
    return knn[:, k_eff - 1]


print(f"\n=== {args.name} ===")

level_stats = {}  # level -> {delta_f_motif, delta_f_bg, score_motif, score_bg, mag_motif, mag_bg}

for batch_idx, batch in enumerate(loader):
    if batch_idx >= args.n_batches:
        break
    if not ('pos' in batch and 'neg' in batch):
        continue

    pos = batch['pos'].to(device)
    neg = batch['neg'].to(device)

    with torch.no_grad():
        xx_pos, pp_pos, _, _, _, internals_pos = model(
            pos.x.float(), pos.edge_index, pos.batch
        )
        _, _, _, _, _, internals_neg = model(
            neg.x.float(), neg.edge_index, neg.batch
        )

        # is_motif gives per-node labels (0=bg, 1=motif)
        y = pos.is_motif.to(device).long()  # [n_nodes] labels at original resolution

        n_levels = min(len(internals_pos), len(internals_neg), len(pp_pos))

        for t in range(1, n_levels):
            x_pos = F.normalize(internals_pos[t]['x_merged'], dim=-1)
            x_neg = F.normalize(internals_neg[t]['x_merged'], dim=-1)
            s = pp_pos[t].detach()

            # map original node labels to supernodes at level t
            cum_assign = internals_pos[t]['cum_assign'].to(device)  # [n_orig] -> supernode id
            n_super = int(cum_assign.max().item()) + 1

            # supernode label: 1 if any original node in it is motif
            super_y = torch.zeros(n_super, dtype=torch.long, device=device)
            super_y.scatter_(0, cum_assign, y)
            # clamp in case scatter with 0 overwrites 1
            motif_mask = torch.zeros(n_super, device=device)
            motif_mask.scatter_(0, cum_assign[y == 1], torch.ones((y == 1).sum(), device=device))
            is_motif = motif_mask > 0

            if x_pos.size(0) < 2 or x_neg.size(0) < 1:
                continue
            k_eff = min(args.k, x_pos.size(0) - 1, x_neg.size(0))
            if k_eff < 1:
                continue

            rho_pos = knn_radius(x_pos, x_pos, k_eff)
            rho_neg = knn_radius(x_pos, x_neg, k_eff)
            delta_f = rho_pos - rho_neg

            # compute transform input magnitudes for the previous level's embeddings
            # feats feeding into pool_layers[t-1] is internals[t-1]['x_merged']
            x_prev = internals_pos[t - 1]['x_merged']
            prev_cum = internals_pos[t - 1]['cum_assign'].to(device)
            n_prev = int(prev_cum.max().item()) + 1
            # map original y to level t-1 supernodes
            prev_motif_mask = torch.zeros(n_prev, device=device)
            prev_motif_mask.scatter_(0, prev_cum[y == 1], torch.ones((y == 1).sum(), device=device))
            prev_is_motif = prev_motif_mask > 0
            mag = x_prev.norm(dim=-1)

            if t not in level_stats:
                level_stats[t] = {
                    'df_motif': [], 'df_bg': [],
                    'score_motif': [], 'score_bg': [],
                    'mag_motif': [], 'mag_bg': [],
                    's_opt_motif': [], 's_opt_bg': [],
                }

            if is_motif.any():
                level_stats[t]['df_motif'].append(delta_f[is_motif].cpu())
                level_stats[t]['score_motif'].append(s[is_motif].cpu())
                level_stats[t]['s_opt_motif'].append(
                    (torch.exp(-args.beta * delta_f[is_motif].clamp(-10, 10)) / (2 * args.lam)).cpu()
                )
            if (~is_motif).any():
                level_stats[t]['df_bg'].append(delta_f[~is_motif].cpu())
                level_stats[t]['score_bg'].append(s[~is_motif].cpu())
                level_stats[t]['s_opt_bg'].append(
                    (torch.exp(-args.beta * delta_f[~is_motif].clamp(-10, 10)) / (2 * args.lam)).cpu()
                )
            if prev_is_motif.any():
                level_stats[t]['mag_motif'].append(mag[prev_is_motif].cpu())
            if (~prev_is_motif).any():
                level_stats[t]['mag_bg'].append(mag[~prev_is_motif].cpu())


for t in sorted(level_stats.keys()):
    st = level_stats[t]

    def _fmt(tensors, name):
        if not tensors:
            return f"{name}: empty"
        v = torch.cat(tensors)
        return f"{name}: mean={v.mean():.3f}  std={v.std():.3f}  min={v.min():.3f}  max={v.max():.3f}  n={v.numel()}"

    print(f"\nlevel {t}:")
    print(f"  delta_f   | {_fmt(st['df_motif'], 'motif')}  |  {_fmt(st['df_bg'], 'bg')}")
    print(f"  score     | {_fmt(st['score_motif'], 'motif')}  |  {_fmt(st['score_bg'], 'bg')}")
    print(f"  s_opt     | {_fmt(st['s_opt_motif'], 'motif')}  |  {_fmt(st['s_opt_bg'], 'bg')}")
    print(f"  input_mag | {_fmt(st['mag_motif'], 'motif')}  |  {_fmt(st['mag_bg'], 'bg')}")
