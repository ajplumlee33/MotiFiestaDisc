"""profile a MotiFiestaDisc forward pass, timing each section.

mps ops are async — call synchronize() before every timer read so numbers
reflect actual gpu time, not just kernel submission time.

run from repo root:
    python scripts/profile_disc.py --name barbell_gin1 --data data/synth-barbell-k10
"""
import argparse
import json
import time

import torch

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.training.loading import get_loader
from MotiFiesta.disc.disc_model import MotiFiestaDisc


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def tick():
    sync()
    return time.perf_counter()


def tock(t0, label, times):
    sync()
    dt = (time.perf_counter() - t0) * 1000
    times[label] = times.get(label, 0.0) + dt
    return time.perf_counter()


def forward_timed(model, x, edge_index, batch, device):
    from torch_geometric.utils import remove_self_loops, coalesce
    times = {}

    t = tick()
    n = x.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes=n)
    x_aug = model._wl_augment(x.float(), edge_index)
    t = tock(t, 'wl_augment', times)

    adj = model._build_adj(edge_index, n)
    t = tock(t, 'build_adj', times)

    for lvl, wl in enumerate(model.walk_lens):
        t = tick()
        subgraph_list = model._sample_subgraphs(adj, batch, walk_len=wl)
        t = tock(t, f'sample_lvl{lvl}', times)

        # ---- embed timing broken into sections ----
        if not subgraph_list:
            continue

        n_sub = len(subgraph_list)
        all_ei_src, all_ei_dst = [], []
        all_anchor_u, all_anchor_v = [], []
        all_sub_assign, all_flat_nodes = [], []
        node_offset = 0

        t = tick()
        for s_idx, (anchor_u, anchor_v, nodes) in enumerate(subgraph_list):
            n_local = len(nodes)
            node_set = set(nodes)
            local_idx = {v: i for i, v in enumerate(nodes)}
            for v in nodes:
                lv = local_idx[v]
                for w in adj[v]:
                    if w in node_set:
                        all_ei_src.append(lv + node_offset)
                        all_ei_dst.append(local_idx[w] + node_offset)
            all_anchor_u.append(local_idx[anchor_u] + node_offset)
            all_anchor_v.append(local_idx[anchor_v] + node_offset)
            all_sub_assign.extend([s_idx] * n_local)
            all_flat_nodes.extend(nodes)
            node_offset += n_local
        t = tock(t, f'pass1_loop_lvl{lvl}', times)

        total_nodes = node_offset

        t = tick()
        flat_nodes_t = torch.tensor(all_flat_nodes, dtype=torch.long, device=device)
        sub_t        = torch.tensor(all_sub_assign, dtype=torch.long, device=device)
        anchor_u_t   = torch.tensor(all_anchor_u,   dtype=torch.long, device=device)
        if all_ei_src:
            ei = torch.tensor([all_ei_src, all_ei_dst], dtype=torch.long, device=device)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long, device=device)
        t = tock(t, f'list_to_tensor_lvl{lvl}', times)

        t = tick()
        reachable = torch.zeros(total_nodes, dtype=torch.long, device=device)
        upd_mask  = torch.empty(total_nodes, dtype=torch.bool,  device=device)
        dist_mask = torch.empty(total_nodes, dtype=torch.bool,  device=device)

        def _bfs(anchor_t):
            dist = torch.full((total_nodes,), 3, dtype=torch.long, device=device)
            dist[anchor_t] = 0
            if ei.size(1) == 0:
                return dist
            ei_src, ei_dst = ei[0], ei[1]
            for hop in range(1, 3):
                reachable.zero_()
                reachable.scatter_add_(0, ei_dst, (dist[ei_src] == hop - 1).long())
                torch.gt(reachable, 0, out=upd_mask)
                torch.gt(dist, hop,  out=dist_mask)
                upd_mask.logical_and_(dist_mask)
                dist.masked_fill_(upd_mask, hop)
            return dist

        dist_u = _bfs(anchor_u_t)
        if model.pair_sampling:
            anchor_v_t = torch.tensor(all_anchor_v, dtype=torch.long, device=device)
            dist_v = _bfs(anchor_v_t)
        t = tock(t, f'bfs_lvl{lvl}', times)

        t = tick()
        dist_dim = 8 if model.pair_sampling else 4
        dist_onehot = torch.zeros(total_nodes, dist_dim, device=device)
        dist_onehot.scatter_(1, dist_u.unsqueeze(1), 1.0)
        if model.pair_sampling:
            dist_onehot.scatter_(1, (dist_v + 4).unsqueeze(1), 1.0)
        X_wl = x_aug[flat_nodes_t].float()
        X = torch.cat([X_wl, dist_onehot], dim=-1)
        t = tock(t, f'feature_assemble_lvl{lvl}', times)

        t = tick()
        H = X
        for mlp in model.gin:
            if ei.size(1) > 0:
                agg = torch.zeros_like(H)
                agg.scatter_add_(0, ei[1].unsqueeze(1).expand(-1, H.size(1)), H[ei[0]])
            else:
                agg = torch.zeros_like(H)
            H = mlp(H + agg)
        t = tock(t, f'gin_lvl{lvl}', times)

        t = tick()
        z_sub = torch.zeros(n_sub, model.hidden_dim, device=device)
        counts = torch.zeros(n_sub, device=device)
        z_sub.scatter_add_(0, sub_t.unsqueeze(1).expand(-1, model.hidden_dim), H)
        counts.scatter_add_(0, sub_t, torch.ones(H.size(0), device=device))
        z_sub = z_sub / counts.unsqueeze(1).clamp(min=1)
        t = tock(t, f'pool_lvl{lvl}', times)

        print(f'  lvl{lvl}: n_sub={n_sub:4d}  total_nodes={total_nodes:5d}  '
              f'n_edges={len(all_ei_src):6d}')

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',     required=True)
    parser.add_argument('--data',     required=True)
    parser.add_argument('--n-graphs', type=int, default=5)
    parser.add_argument('--warmup',   type=int, default=2)
    args = parser.parse_args()

    device = get_device()

    ckpt = torch.load(f'models/{args.name}/{args.name}_best.pth',
                      map_location='cpu', weights_only=False)
    with open(f'models/{args.name}/hparams.json') as f:
        hp = json.load(f)['model']

    wl_raw = hp.get('walk_lens', hp.get('walk_len', 8))
    walk_lens = [int(x) for x in wl_raw.split(',')] if isinstance(wl_raw, str) else (
        wl_raw if isinstance(wl_raw, list) else [wl_raw])

    model = MotiFiestaDisc(
        n_features=hp['n_features'],
        hidden_dim=hp.get('hidden_dim', 32),
        gin_layers=hp.get('gin_layers', 2),
        walk_lens=walk_lens,
        n_walks=hp.get('n_walks', 4),
        wl_hops=hp.get('wl_hops', 1),
        pair_sampling=hp.get('pair_sampling', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    dataset = get_loader(root=args.data, name='synth_pairs')

    graphs = []
    for idx, g_pair in enumerate(dataset['dataset_whole']):
        if idx >= args.n_graphs + args.warmup:
            break
        g = g_pair['pos']
        n = len(g.x)
        batch = torch.zeros(n, dtype=torch.long, device=device)
        graphs.append((g.x.float().to(device), g.edge_index.to(device), batch))

    print(f'\nmodel: {args.name}  device: {device}  pair_sampling: {model.pair_sampling}')
    print(f'walk_lens: {model.walk_lens}  hidden_dim: {model.hidden_dim}\n')

    # warmup
    print(f'warming up ({args.warmup} graphs)...')
    with torch.no_grad():
        for x, ei, b in graphs[:args.warmup]:
            _ = forward_timed(model, x, ei, b, device)

    # timed runs
    print(f'\nprofiling ({args.n_graphs} graphs)...\n')
    totals = {}
    with torch.no_grad():
        for i, (x, ei, b) in enumerate(graphs[args.warmup:]):
            print(f'--- graph {i} ---')
            t = forward_timed(model, x, ei, b, device)
            for k, v in t.items():
                totals[k] = totals.get(k, 0.0) + v

    n = args.n_graphs
    print(f'\n{"section":<30}  {"total ms":>10}  {"per graph ms":>14}')
    print('-' * 58)
    grand = 0.0
    for k, v in totals.items():
        print(f'{k:<30}  {v:>10.1f}  {v/n:>14.2f}')
        grand += v
    print(f'{"TOTAL":<30}  {grand:>10.1f}  {grand/n:>14.2f}')


if __name__ == '__main__':
    main()
