"""
evaluate a MotiFiestaDisc model.

reports two metrics:
  subgraph jaccard: score-ranked ego-nets vs ground truth, no lsh
  lsh jaccard:      table1 sweep — hash_dim in [8,16,32], top_k=3, n_runs=3, n_graphs=200

run from repo root:
    python MotiFiesta/disc/disc_decode.py --name barbell_ego1 --data data/synth-barbell-k10
"""
import argparse
import statistics
from itertools import permutations

import torch
import torch.nn.functional as F
from lshashpy3 import LSHash

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.training.loading import get_loader


def _embed_all(model, dataset, device, n_graphs=-1):
    """run model once per graph; cache per-level dicts."""
    cache = []
    model.eval()
    for idx, g_pair in enumerate(dataset['dataset_whole']):
        if n_graphs > -1 and idx > n_graphs:
            break
        g = g_pair['pos']
        n_nodes = len(g.x)
        batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            levels = model(g.x.float().to(device), g.edge_index.to(device), batch)
        cache.append({
            'levels':   [{k: v.cpu() for k, v in lvl.items() if isinstance(v, torch.Tensor)}
                         for lvl in levels],
            'n_nodes':  n_nodes,
            'motif_id': g.motif_id,
        })
    return cache


def _eval_subgraph_jaccard(cache, level, top_k=3):
    """score-ranked subgraph jaccard: no lsh, no node labeling.

    takes the top_k scoring ego-nets, unions their node sets, and computes
    jaccard against ground-truth motif_id. measures how well score alone
    identifies the motif — an upper bound on what lsh decode can achieve.
    """
    jacc_vals = []
    for entry in cache:
        if level >= len(entry['levels']):
            continue
        lvl        = entry['levels'][level]
        scores     = lvl['scores']       # (n_subs,)
        flat_nodes = lvl['flat_nodes']   # (total_flat,)
        flat_subs  = lvl['flat_subs']    # (total_flat,)
        motif_id   = entry['motif_id']   # (n_nodes,)

        n_subs = scores.size(0)
        if n_subs == 0 or flat_nodes.numel() == 0:
            continue

        sub_node_sets = [set() for _ in range(n_subs)]
        for node, sub in zip(flat_nodes.tolist(), flat_subs.tolist()):
            sub_node_sets[sub].add(node)

        ranked = scores.argsort(descending=True).tolist()
        pred_nodes = set()
        for sub_i in ranked[:top_k]:
            pred_nodes.update(sub_node_sets[sub_i])

        true_nodes = set(motif_id.nonzero(as_tuple=False).squeeze(-1).tolist())
        if not true_nodes:
            continue

        inter = len(pred_nodes & true_nodes)
        union = len(pred_nodes | true_nodes)
        jacc_vals.append(inter / union if union > 0 else 0.0)

    return statistics.mean(jacc_vals) if jacc_vals else 0.0


def _decode_from_cache(model, cache, hash_dim, level, dummy=False):
    """lsh hash cached embeddings; scatter to nodes via node_to_sub."""
    hash_table = LSHash(hash_dim, model.hidden_dim)
    hash_set = set()
    results = []
    skipped = set()
    spot_count = 0

    for idx, entry in enumerate(cache):
        if level >= len(entry['levels']):
            skipped.add(idx)
            results.append(None)
            continue

        lvl     = entry['levels'][level]
        n_nodes = entry['n_nodes']
        Z       = lvl['z_sub']
        scores  = lvl['scores']
        mh      = lvl['node_to_sub']

        if dummy:
            Z      = torch.randn_like(Z)
            scores = torch.rand_like(scores)

        hashes = [hash_table.index(Z[i].numpy())[0] for i in range(len(Z))]
        hash_set.update(hashes)

        node_scores   = scores[mh]
        sub_spot_ids  = torch.arange(len(Z)) + spot_count
        spotlight_ids = sub_spot_ids[mh]
        node_hashes   = [hashes[mh[v].item()] for v in range(n_nodes)]
        spot_count   += len(Z)

        results.append({'hashes': node_hashes, 'scores': node_scores, 'spot': spotlight_ids})

    hash_idx = {h: i + 1 for i, h in enumerate(sorted(hash_set))}
    decoded = []
    for idx, entry in enumerate(cache):
        if idx in skipped:
            continue
        r = results[idx]
        decoded.append({
            'motif_id':     entry['motif_id'],
            'cum_scores':   r['scores'],
            'motif_pred':   torch.tensor([hash_idx[h] for h in r['hashes']]),
            'spotlight_ids': r['spot'],
        })
    return decoded


def _eval_lsh(decoded_graphs, n_motifs=1, top_k=1):
    sigma_all       = torch.cat([g['cum_scores'] for g in decoded_graphs])
    motifs_pred_all = torch.cat([g['motif_pred'] for g in decoded_graphs])
    true_motif_ids  = torch.cat([g['motif_id']   for g in decoded_graphs])

    motifs_input = torch.unique(motifs_pred_all)
    motif_indices = {m.item(): i for i, m in enumerate(motifs_input)}
    for i in range(len(motifs_pred_all)):
        motifs_pred_all[i] = motif_indices[motifs_pred_all[i].item()]

    motif_ids, counts = torch.unique(motifs_pred_all, return_counts=True)
    rank_scores = torch.zeros_like(motif_ids, dtype=torch.float32).scatter_add(
        0, motifs_pred_all, sigma_all) / counts
    motifs_sorted = torch.argsort(rank_scores, descending=True)
    ranks = torch.zeros_like(motifs_sorted)
    for ind, val in enumerate(motifs_sorted):
        ranks[val] = ind

    motifs_pred_all = torch.where(ranks[motifs_pred_all] < top_k, motifs_pred_all + 1, 0)
    pred = F.one_hot(motifs_pred_all)
    non_empty_mask = pred.abs().sum(dim=0).bool()
    pred = pred[:, non_empty_mask]
    true = F.one_hot(true_motif_ids)[:, 1:]

    best_jaccard = 0.0
    for p in permutations(range(pred.shape[1])):
        p = torch.tensor(p)
        pred_slice = pred[:, p][:, :n_motifs]
        num = torch.min(pred_slice, true).sum(dim=0)
        den = torch.max(pred_slice, true).sum(dim=0)
        jaccard = (num / den).sum().item()
        if jaccard > best_jaccard:
            best_jaccard = jaccard
    return best_jaccard


def eval_config(model, dataset, device, hash_dim, level, top_k, n_runs, n_graphs,
                cache=None):
    """evaluate one (hash_dim, level) config with lsh. pass pre-built cache to reuse embeddings."""
    if cache is None:
        cache = _embed_all(model, dataset, device, n_graphs)
    scores = []
    for _ in range(n_runs):
        dg = _decode_from_cache(model, cache, hash_dim, level)
        scores.append(_eval_lsh(dg, n_motifs=1, top_k=top_k))
    mean = statistics.mean(scores)
    std  = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return mean, std


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',     required=True)
    parser.add_argument('--data',     required=True)
    parser.add_argument('--dataset',  type=str, default='synth_pairs')
    parser.add_argument('--n-runs',   type=int, default=3)
    parser.add_argument('--n-graphs', type=int, default=200)
    parser.add_argument('--top-k',    type=int, default=3)
    parser.add_argument('--quick',    action='store_true',
                        help='fast sanity check: hash_dim=16, level=0, n_graphs=30, n_runs=1')
    args = parser.parse_args()

    if args.quick:
        args.n_graphs = 30
        args.n_runs   = 1

    device = get_device()

    import json
    from MotiFiesta.disc.disc_model import MotiFiestaDisc
    ckpt = torch.load(f'models/{args.name}/{args.name}_best.pth', map_location='cpu',
                      weights_only=False)

    with open(f'models/{args.name}/hparams.json') as f:
        hp = json.load(f)['model']

    wl_raw    = hp.get('walk_lens', [1, 2, 3])
    walk_lens = [int(x) for x in wl_raw.split(',')] if isinstance(wl_raw, str) else (
                wl_raw if isinstance(wl_raw, list) else [wl_raw])

    model = MotiFiestaDisc(
        n_features = hp['n_features'],
        hidden_dim = hp.get('hidden_dim', 32),
        gin_layers = hp.get('gin_layers', 2),
        wl_hops    = hp.get('wl_hops', 1),
        walk_lens  = walk_lens,
        pool       = hp.get('pool', 'mean'),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    dataset  = get_loader(root=args.data, name=args.dataset)
    n_levels = len(walk_lens)
    hash_dims = [16]              if args.quick else [8, 16, 32]
    levels    = [0]               if args.quick else list(range(n_levels))

    print(f"\n=== {args.name} (top_k={args.top_k}, n_runs={args.n_runs}, "
          f"n_graphs={args.n_graphs}) ===\n")
    print("embedding graphs...", flush=True)
    cache = _embed_all(model, dataset, device, args.n_graphs)
    print(f"cached {len(cache)} graphs\n", flush=True)

    print("subgraph jaccard (score-ranked, no lsh):")
    for level in levels:
        sj = _eval_subgraph_jaccard(cache, level, top_k=args.top_k)
        print(f"  level={level} (hop={walk_lens[level]}):  {sj:.4f}")
    print()

    print(f"lsh jaccard (hash_dim∈{hash_dims} × level∈{levels}):")
    best = {'mean': 0.0, 'std': 0.0, 'hash_dim': None, 'level': None}
    for hash_dim in hash_dims:
        for level in levels:
            mean, std = eval_config(model, dataset, device, hash_dim, level,
                                    args.top_k, args.n_runs, args.n_graphs, cache=cache)
            marker = ' *' if mean > best['mean'] else ''
            print(f"  hash_dim={hash_dim}  level={level}:  {mean:.4f} ±{std:.4f}{marker}")
            if mean > best['mean']:
                best = {'mean': mean, 'std': std, 'hash_dim': hash_dim, 'level': level}

    dummy_scores = []
    for _ in range(args.n_runs):
        dg = _decode_from_cache(model, cache, best['hash_dim'], best['level'], dummy=True)
        dummy_scores.append(_eval_lsh(dg, n_motifs=1, top_k=args.top_k))
    dm = statistics.mean(dummy_scores)
    ds = statistics.stdev(dummy_scores) if len(dummy_scores) > 1 else 0.0

    print(f"\nbest lsh:  {best['mean']:.2f} ±{best['std']:.2f}  "
          f"(hash_dim={best['hash_dim']}, level={best['level']})")
    print(f"dummy:     {dm:.2f} ±{ds:.2f}")
    print(f"\nfinal: {best['mean']:.2f} ±{best['std']:.2f}  ({dm:.2f} ±{ds:.2f})")


if __name__ == '__main__':
    main()
    import sys; sys.exit(0)
