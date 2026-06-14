#!/usr/bin/env python3
"""diagnose disc model: score gap and embedding cos_sim per level.

usage:
    python scripts/diag_disc.py --name barbell_rwr1 --data data/synth-barbell-k10
"""
import argparse
import json

import torch
import torch.nn.functional as F

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.training.loading import get_loader
from MotiFiesta.training.disc_model import MotiFiestaDisc


def load_model(name, device):
    ckpt = torch.load(f'models/{name}/{name}_best.pth', map_location='cpu', weights_only=False)
    with open(f'models/{name}/hparams.json') as f:
        hp = json.load(f)['model']
    wl_raw = hp.get('walk_lens', hp.get('walk_len', 8))
    walk_lens = [int(x) for x in wl_raw.split(',')] if isinstance(wl_raw, str) else (
        wl_raw if isinstance(wl_raw, list) else [wl_raw])
    model = MotiFiestaDisc(
        n_features=hp['n_features'],
        rwse_steps=hp.get('rwse_steps', 8),
        walk_lens=walk_lens,
        n_walks=hp.get('n_walks', 4),
        wl_hops=hp.get('wl_hops', 1),
        pair_sampling=hp.get('pair_sampling', False),
        rwr_alpha=hp.get('rwr_alpha', 0.0),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',     required=True)
    parser.add_argument('--data',     required=True)
    parser.add_argument('--n-graphs', type=int, default=100)
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.name, device)
    dataset = get_loader(root=args.data, name='synth_pairs')

    # collect z_sub, scores, and motif_id labels per level
    level_z   = [[] for _ in model.walk_lens]
    level_s   = [[] for _ in model.walk_lens]
    level_lab = [[] for _ in model.walk_lens]   # 1=motif node, 0=background
    level_wts = [[] for _ in model.walk_lens]   # score_net weights for interpretation

    with torch.no_grad():
        for idx, g_pair in enumerate(dataset['dataset_whole']):
            if idx >= args.n_graphs:
                break
            g = g_pair['pos']
            n = len(g.x)
            batch = torch.zeros(n, dtype=torch.long, device=device)
            embs, probas, _, _, merge_info, internals = model(
                g.x.float().to(device), g.edge_index.to(device), batch
            )
            motif_id = g.motif_id.to(device)   # (n_nodes,) 0/1

            for lvl in range(len(model.walk_lens)):
                z   = internals[lvl]['z_sub']          # (n_sub, z_dim)
                s   = internals[lvl]['scores']         # (n_sub,)
                mh  = internals[lvl]['node_to_sub']    # (n_nodes,) node→sub index

                # label each subgraph: majority motif_id among its member nodes
                n_sub = z.size(0)
                sub_motif_count = torch.zeros(n_sub, device=device)
                sub_node_count  = torch.zeros(n_sub, device=device)
                sub_motif_count.scatter_add_(0, mh, motif_id.float())
                sub_node_count.scatter_add_(0, mh, torch.ones(n, device=device))
                sub_label = (sub_motif_count / sub_node_count.clamp(min=1) > 0.5).long()

                level_z[lvl].append(z.cpu())
                level_s[lvl].append(s.cpu())
                level_lab[lvl].append(sub_label.cpu())

    print(f'\n=== {args.name} ===\n')
    print(f'{"lvl":>4}  {"wl":>4}  {"s_motif":>8}  {"s_bg":>8}  {"gap":>8}'
          f'  {"cos_mm":>8}  {"cos_bb":>8}  {"cos_mb":>8}  {"n_mot":>6}  {"n_bg":>6}')
    print('-' * 90)

    for lvl, wl in enumerate(model.walk_lens):
        Z   = torch.cat(level_z[lvl],   dim=0)   # (N_sub, z_dim)
        S   = torch.cat(level_s[lvl],   dim=0)   # (N_sub,)
        lab = torch.cat(level_lab[lvl], dim=0)   # (N_sub,)

        mot_mask = lab == 1
        bg_mask  = lab == 0
        n_mot = mot_mask.sum().item()
        n_bg  = bg_mask.sum().item()

        s_mot = S[mot_mask].mean().item() if n_mot > 0 else float('nan')
        s_bg  = S[bg_mask].mean().item()  if n_bg  > 0 else float('nan')
        gap   = s_mot - s_bg

        # cosine similarity matrices (sampled to avoid OOM)
        def sample(mask, k=300):
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.size(0) > k:
                idx = idx[torch.randperm(idx.size(0))[:k]]
            return F.normalize(Z[idx], dim=-1)

        Zm = sample(mot_mask)
        Zb = sample(bg_mask)

        cos_mm = (Zm @ Zm.T).fill_diagonal_(float('nan')).nanmean().item() if Zm.size(0) > 1 else float('nan')
        cos_bb = (Zb @ Zb.T).fill_diagonal_(float('nan')).nanmean().item() if Zb.size(0) > 1 else float('nan')
        cos_mb = (Zm @ Zb.T).mean().item() if (Zm.size(0) > 0 and Zb.size(0) > 0) else float('nan')

        print(f'{lvl:>4}  {wl:>4}  {s_mot:>8.3f}  {s_bg:>8.3f}  {gap:>8.3f}'
              f'  {cos_mm:>8.4f}  {cos_bb:>8.4f}  {cos_mb:>8.4f}  {n_mot:>6}  {n_bg:>6}')

    # score_net weights for level 2 (most informative)
    print('\n--- score_net weights (level 2) ---')
    w = model.score_nets[-1].weight.data.squeeze()
    top_pos = w.topk(6).indices.tolist()
    top_neg = (-w).topk(6).indices.tolist()
    print(f'  + bins (x[i] if i<{model.n_features}, else nbr_sum[i-{model.n_features}]): {top_pos}')
    print(f'  - bins: {top_neg}')


if __name__ == '__main__':
    main()
