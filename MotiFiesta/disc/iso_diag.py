"""evaluate embedding quality by subgraph isomorphism class.

uses weisfeiler-lehman graph hash as a structural label (no node features).
checks whether z_sub clusters by isomorphism class independent of motif labels.

usage:
    python MotiFiesta/disc/iso_diag.py --name proteins_ns10 --dataset PROTEINS --n-graphs 200
"""
import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import networkx as nx

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.training.loading import get_loader
from MotiFiesta.disc.disc_model import MotiFiestaDisc


def load_model(name, device):
    ckpt = torch.load(f'models/{name}/{name}_best.pth', map_location='cpu', weights_only=False)
    with open(f'models/{name}/hparams.json') as f:
        hp = json.load(f)['model']
    model = MotiFiestaDisc(
        n_features=hp['n_features'],
        hidden_dim=hp.get('hidden_dim', 64),
        gin_layers=hp.get('gin_layers', 2),
        k_max=hp.get('k_max', 12),
        n_samples=hp.get('n_samples', 50),
        wl_hops=hp.get('wl_hops', 0),
        pool=hp.get('pool', 'mean'),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',           required=True)
    parser.add_argument('--dataset',        default='PROTEINS')
    parser.add_argument('--n-graphs',       type=int, default=200)
    parser.add_argument('--min-class-size', type=int, default=20,
                        help='min subgraphs per isomorphism class to include')
    parser.add_argument('--top-k',          type=int, default=10,
                        help='top-k most frequent classes to report')
    args = parser.parse_args()

    device  = get_device()
    model   = load_model(args.name, device)
    dataset = get_loader(root=f'data/{args.dataset}', name=args.dataset)

    all_z, all_s, all_hashes = [], [], []

    with torch.no_grad():
        for idx, g_pair in enumerate(dataset['dataset_whole']):
            if idx >= args.n_graphs:
                break
            g     = g_pair['pos']
            n     = len(g.x)
            batch = torch.zeros(n, dtype=torch.long, device=device)
            levels = model(g.x.float().to(device), g.edge_index.to(device), batch)

            lvl     = levels[0]
            z       = lvl['z_sub']
            s       = lvl['scores']
            sg_data = lvl['sg_data']

            for n_nodes, edges, _ in sg_data:
                G = nx.Graph()
                G.add_nodes_from(range(n_nodes))
                G.add_edges_from(edges)
                all_hashes.append(nx.weisfeiler_lehman_graph_hash(G))

            all_z.append(z.cpu())
            all_s.append(s.cpu())

    Z   = torch.cat(all_z, dim=0)
    S   = torch.cat(all_s, dim=0)
    Z_n = F.normalize(Z, dim=-1)

    class_indices = defaultdict(list)
    for i, h in enumerate(all_hashes):
        class_indices[h].append(i)

    classes        = {h: idxs for h, idxs in class_indices.items() if len(idxs) >= args.min_class_size}
    classes_sorted = sorted(classes.items(), key=lambda x: -len(x[1]))

    print(f'\n=== {args.name} — isomorphism class analysis (n_graphs={args.n_graphs}) ===\n')
    print(f'total subgraphs : {len(all_hashes)}')
    print(f'unique classes  : {len(class_indices)}')
    print(f'classes ≥{args.min_class_size}     : {len(classes)}\n')

    print(f'{"class hash":<14} {"count":>6} {"s_mean":>8} {"s_std":>7} {"cos_within":>11}')
    print('-' * 52)

    within_sims, between_sims = [], []
    top = classes_sorted[:args.top_k]

    for h, idxs in top:
        idx_t = torch.tensor(idxs)
        Zc    = Z_n[idx_t]
        Sc    = S[idx_t]
        cw    = (Zc @ Zc.T).fill_diagonal_(float('nan')).nanmean().item() if len(idxs) > 1 else float('nan')
        within_sims.append(cw)
        print(f'{h[:12]:<14} {len(idxs):>6} {Sc.mean().item():>8.3f} {Sc.std().item():>7.3f} {cw:>11.4f}')

    for i, (h1, idxs1) in enumerate(top):
        Z1 = Z_n[torch.tensor(idxs1[:100])]
        for _, idxs2 in top[i+1:]:
            Z2 = Z_n[torch.tensor(idxs2[:100])]
            between_sims.append((Z1 @ Z2.T).mean().item())

    if within_sims and between_sims:
        mean_w = sum(w for w in within_sims if w == w) / sum(1 for w in within_sims if w == w)
        mean_b = sum(between_sims) / len(between_sims)
        print(f'\nwithin-class cos  : {mean_w:.4f}')
        print(f'between-class cos : {mean_b:.4f}')
        print(f'separation gap    : {mean_w - mean_b:.4f}')


if __name__ == '__main__':
    main()
