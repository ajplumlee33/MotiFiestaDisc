"""
evaluate a MotiFiestaDisc model with the table1 sweep:
  hash_dim in [8, 16, 32], level in [0, 1, 2], top_k=3, n_runs=3, n_graphs=200

run from repo root:
    python scripts/disc_decode.py --name barbell_disc1 --data data/synth-barbell-k10
"""
import argparse
import statistics
from itertools import permutations

import torch
import torch.nn.functional as F
from lshashpy3 import LSHash

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.training.loading import get_loader


class _GraphResult:
    """lightweight container for decoded graph results."""
    def __init__(self, motif_id, cum_scores, motif_pred, spotlight_ids):
        self.motif_id = motif_id
        self.cum_scores = cum_scores
        self.motif_pred = motif_pred
        self.spotlight_ids = spotlight_ids


def _embed_all(model, dataset, device, n_graphs=-1):
    """run model once per graph; cache embeddings for all levels."""
    cache = []
    model.eval()
    for idx, g_pair in enumerate(dataset['dataset_whole']):
        if n_graphs > -1 and idx > n_graphs:
            break
        g = g_pair['pos']
        n_nodes = len(g.x)
        batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            embs, probas, _, _, merge_info, _ = model(
                g.x.float().to(device), g.edge_index.to(device), batch
            )
        levels = {}
        for lvl in range(len(embs)):
            levels[lvl] = {
                'Z':      embs[lvl].cpu(),
                'scores': probas[lvl].cpu(),
                'mh':     merge_info['node_to_sub'][lvl].cpu(),
            }
        cache.append({'levels': levels, 'n_nodes': n_nodes, 'motif_id': g.motif_id})
    return cache


def _decode_from_cache(model, cache, hash_dim, level, dummy=False):
    """hash cached embeddings without re-running the model.
    lsh is re-initialized each call, giving different random projections per run.
    """
    hash_table = LSHash(hash_dim, model.hidden_dim)
    hash_set = set()
    results = []
    skipped = set()
    spot_count = 0

    for idx, entry in enumerate(cache):
        if level not in entry['levels']:
            skipped.add(idx)
            results.append(None)
            continue

        lc = entry['levels'][level]
        n_nodes = entry['n_nodes']
        Z = lc['Z']
        sup_scores = lc['scores']
        mh = lc['mh']

        if dummy:
            Z = torch.randn_like(Z)
            sup_scores = torch.rand_like(sup_scores)

        # hash all subgraphs, then scatter to nodes via mh
        hashes_for_subs = [hash_table.index(Z[i].numpy())[0] for i in range(len(Z))]
        hash_set.update(hashes_for_subs)

        motif_scores = sup_scores[mh]                          # (n_nodes,)
        sub_spot_ids = torch.arange(len(Z)) + spot_count
        spotlight_ids = sub_spot_ids[mh]                       # (n_nodes,)
        g_hashes = [hashes_for_subs[mh[node].item()] for node in range(n_nodes)]
        spot_count += len(Z)

        results.append({'hashes': g_hashes, 'scores': motif_scores, 'spot': spotlight_ids})

    hash_idx = {h: i + 1 for i, h in enumerate(sorted(hash_set))}
    decoded = []
    for idx, entry in enumerate(cache):
        if idx in skipped:
            continue
        r = results[idx]
        motif_pred = torch.tensor([hash_idx[h] for h in r['hashes']])
        decoded.append(_GraphResult(
            motif_id=entry['motif_id'],
            cum_scores=r['scores'],
            motif_pred=motif_pred,
            spotlight_ids=r['spot'],
        ))
    return decoded


def _decode(model, dataset, device, hash_dim, level, dummy=False, n_graphs=-1):
    """single-shot decode. builds cache internally; kept for table1_disc compat."""
    cache = _embed_all(model, dataset, device, n_graphs)
    return _decode_from_cache(model, cache, hash_dim, level, dummy=dummy)


def _eval(decoded_graphs, n_motifs=1, top_k=1):
    sigma_all = torch.cat([g.cum_scores for g in decoded_graphs])
    motifs_pred_all = torch.cat([g.motif_pred for g in decoded_graphs])
    true_motif_ids = torch.cat([g.motif_id for g in decoded_graphs])

    # reindex
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

    best_jaccard = 0
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
    """evaluate one (hash_dim, level) config. pass pre-built cache to avoid re-running model."""
    if cache is None:
        cache = _embed_all(model, dataset, device, n_graphs)
    scores = []
    for _ in range(n_runs):
        dg = _decode_from_cache(model, cache, hash_dim, level)
        scores.append(_eval(dg, n_motifs=1, top_k=top_k))
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
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
        args.n_runs = 1

    device = get_device()

    import json
    from MotiFiesta.disc.disc_model import MotiFiestaDisc
    ckpt = torch.load(f'models/{args.name}/{args.name}_best.pth', map_location='cpu',
                      weights_only=False)
    msd = ckpt['model_state_dict']

    with open(f'models/{args.name}/hparams.json') as f:
        hparams_json = json.load(f)

    n_features_base = hparams_json['model']['n_features']
    rwse_steps = hparams_json['model'].get('rwse_steps', 8)
    n_walks = hparams_json['model'].get('n_walks', 4)
    wl_raw = hparams_json['model'].get('walk_lens', hparams_json['model'].get('walk_len', 8))
    walk_lens = [int(x) for x in wl_raw.split(',')] if isinstance(wl_raw, str) else (
        wl_raw if isinstance(wl_raw, list) else [wl_raw])
    n_levels = len(walk_lens)

    wl_hops       = hparams_json['model'].get('wl_hops', 1)
    hidden_dim    = hparams_json['model'].get('hidden_dim', 32)
    gin_layers    = hparams_json['model'].get('gin_layers', 2)
    pair_sampling = hparams_json['model'].get('pair_sampling', False)
    model = MotiFiestaDisc(n_features=n_features_base,
                           hidden_dim=hidden_dim, gin_layers=gin_layers,
                           rwse_steps=rwse_steps, walk_lens=walk_lens, n_walks=n_walks,
                           wl_hops=wl_hops, pair_sampling=pair_sampling)
    model.load_state_dict(msd)
    model.to(device)
    model.eval()

    dataset = get_loader(root=args.data, name=args.dataset)

    hash_dims = [16] if args.quick else [8, 16, 32]
    levels    = [0]  if args.quick else list(range(n_levels))

    print(f"\n=== {args.name} (hash_dim∈{hash_dims} × level∈{levels}, "
          f"top_k={args.top_k}, n_runs={args.n_runs}, n_graphs={args.n_graphs}) ===\n")

    # run model once; reuse embeddings across all hash_dim/level/run combos
    print("embedding graphs...", flush=True)
    cache = _embed_all(model, dataset, device, args.n_graphs)
    print(f"cached {len(cache)} graphs\n", flush=True)

    best = {'mean': 0.0, 'std': 0.0, 'hash_dim': None, 'level': None}

    for hash_dim in hash_dims:
        for level in levels:
            mean, std = eval_config(model, dataset, device, hash_dim, level,
                                    args.top_k, args.n_runs, args.n_graphs, cache=cache)
            marker = ' *' if mean > best['mean'] else ''
            print(f"  hash_dim={hash_dim}  level={level}:  {mean:.4f} ±{std:.4f}{marker}")
            if mean > best['mean']:
                best = {'mean': mean, 'std': std, 'hash_dim': hash_dim, 'level': level}

    # dummy baseline
    dummy_scores = []
    for _ in range(args.n_runs):
        dg = _decode_from_cache(model, cache, best['hash_dim'], best['level'], dummy=True)
        dummy_scores.append(_eval(dg, n_motifs=1, top_k=args.top_k))
    dm = statistics.mean(dummy_scores)
    ds = statistics.stdev(dummy_scores) if len(dummy_scores) > 1 else 0.0

    print(f"\nbest:  {best['mean']:.2f} ±{best['std']:.2f}  "
          f"(hash_dim={best['hash_dim']}, level={best['level']})")
    print(f"dummy: {dm:.2f} ±{ds:.2f}")
    print(f"\nfinal: {best['mean']:.2f} ±{best['std']:.2f}  ({dm:.2f} ±{ds:.2f})")


if __name__ == '__main__':
    main()
    import sys; sys.exit(0)
