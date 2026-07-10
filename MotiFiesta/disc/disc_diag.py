"""diagnose disc model: embedding cosine similarity by motif/bg label.

usage:
    python MotiFiesta/disc/diag_disc.py --name barbell_wwl2 --data data/synth-barbell-k10
"""
import argparse
import json

import torch
import torch.nn.functional as F

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.training.loading import get_loader
from MotiFiesta.disc.disc_model import MotiFiestaDisc


def load_model(name, device):
    ckpt = torch.load(f'models/{name}/{name}_best.pth', map_location='cpu', weights_only=False)
    with open(f'models/{name}/hparams.json') as f:
        hp = json.load(f)['model']
    model = MotiFiestaDisc(
        n_features = hp['n_features'],
        hidden_dim = hp.get('hidden_dim', 64),
        gin_layers = hp.get('gin_layers', 2),
        gnn_type   = hp.get('gnn_type', 'gin'),
        k_max      = hp.get('k_max', 12),
        n_samples  = hp.get('n_samples', 50),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',     required=True)
    parser.add_argument('--data',     required=True)
    parser.add_argument('--n-graphs', type=int, default=100)
    args = parser.parse_args()

    device = get_device()
    model  = load_model(args.name, device)
    dataset = get_loader(root=args.data, name='synth_pairs')

    all_z, all_lab, all_k = [], [], []

    with torch.no_grad():
        for idx, g_pair in enumerate(dataset['dataset_whole']):
            if idx >= args.n_graphs:
                break
            g        = g_pair['pos']
            n        = len(g.x)
            batch    = torch.zeros(n, dtype=torch.long, device=device)
            levels   = model(g.x.float().to(device), g.edge_index.to(device), batch)
            motif_id = g.motif_id.to(device)

            lvl        = levels[0]
            z          = lvl['z_sub']
            flat_nodes = lvl['flat_nodes']
            flat_subs  = lvl['flat_subs']
            n_sub      = z.size(0)

            sub_motif = torch.zeros(n_sub, device=device)
            sub_count = torch.zeros(n_sub, device=device)
            sub_motif.scatter_add_(0, flat_subs, motif_id[flat_nodes].float())
            sub_count.scatter_add_(0, flat_subs, torch.ones(flat_nodes.size(0), device=device))
            sub_label = (sub_motif / sub_count.clamp(min=1) > 0.5).long()

            all_z.append(z.cpu())
            all_lab.append(sub_label.cpu())
            all_k.append(lvl['k_sizes'].cpu())

    Z   = torch.cat(all_z,   dim=0)
    lab = torch.cat(all_lab, dim=0)
    K   = torch.cat(all_k,   dim=0)

    mot_mask = lab == 1
    bg_mask  = lab == 0
    n_mot = mot_mask.sum().item()
    n_bg  = bg_mask.sum().item()

    def sample(mask, k=500):
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.size(0) > k:
            idx = idx[torch.randperm(idx.size(0))[:k]]
        return F.normalize(Z[idx], dim=-1)

    Zm = sample(mot_mask)
    Zb = sample(bg_mask)

    cos_mm = (Zm @ Zm.T).fill_diagonal_(float('nan')).nanmean().item() if Zm.size(0) > 1 else float('nan')
    cos_bb = (Zb @ Zb.T).fill_diagonal_(float('nan')).nanmean().item() if Zb.size(0) > 1 else float('nan')
    cos_mb = (Zm @ Zb.T).mean().item() if (Zm.size(0) > 0 and Zb.size(0) > 0) else float('nan')

    z_mot_norm = Z[mot_mask].norm(dim=-1).mean().item() if n_mot > 0 else float('nan')
    z_bg_norm  = Z[bg_mask].norm(dim=-1).mean().item()  if n_bg  > 0 else float('nan')

    print(f'\n=== {args.name} (n_graphs={args.n_graphs}) ===\n')
    print(f'subgraphs:  motif={n_mot}  bg={n_bg}  ratio={n_mot/(n_mot+n_bg):.3f}')
    print()
    print(f'z_sub norms (raw):')
    print(f'  motif  {z_mot_norm:.4f}')
    print(f'  bg     {z_bg_norm:.4f}')
    print()
    print(f'cosine similarity (normalized):')
    print(f'  motif-motif  {cos_mm:.4f}  (1.0 = perfect cluster)')
    print(f'  bg-bg        {cos_bb:.4f}')
    print(f'  motif-bg     {cos_mb:.4f}  (0.0 = fully separated)')
    print()
    print(f'cos gap by subgraph size k:')
    for k_val in range(3, K.max().item() + 1):
        km = K == k_val
        if km.sum() < 5:
            continue
        zm_k = sample(km & mot_mask, k=200)
        zb_k = sample(km & bg_mask,  k=200)
        nm = (km & mot_mask).sum().item()
        nb = (km & bg_mask).sum().item()
        if zm_k.size(0) > 1 and zb_k.size(0) > 1:
            mm_k = (zm_k @ zm_k.T).fill_diagonal_(float('nan')).nanmean().item()
            mb_k = (zm_k @ zb_k.T).mean().item()
            gap_k = mm_k - mb_k
        else:
            mm_k = mb_k = gap_k = float('nan')
        print(f'  k={k_val:2d}  n_mot={nm:5d}  n_bg={nb:5d}  '
              f'mm={mm_k:.4f}  mb={mb_k:.4f}  gap={gap_k:+.4f}')
    print()


if __name__ == '__main__':
    main()
