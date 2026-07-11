"""
evaluate a MotiFiestaDisc model.

lsh jaccard: table1 sweep — hash_dim in [8,16,32], top_k=3, n_runs=3, n_graphs=200

buckets (s/m/l by subgraph size) are the analog of pooling levels in main branch decode.
sweep: hash_dim × bucket = 9 configs, same structure as main branch hash_dim × level.

run from repo root:
    python MotiFiesta/disc/disc_decode.py --name barbell_strat6 --data data/synth-barbell-k10
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
    """run model once per graph; cache level-0 tensors."""
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
            'lvl':      {k: v.cpu() for k, v in levels[0].items() if isinstance(v, torch.Tensor)},
            'n_nodes':  n_nodes,
            'motif_id': g.motif_id,
        })
    return cache


def _k_to_bucket(k):
    if k <= 4: return 0
    if k <= 8: return 1
    return 2


def _decode(model, cache, hash_dim, bucket, dummy=False):
    """analog of HashDecoder.decode at one level.

    bucket 0=s (k≤4), 1=m (k5-8), 2=l (k9+) — analog of pooling level.
    one hash table for all graphs in this bucket.
    each node assigned to its highest-scoring subgraph in the bucket.
    uncovered nodes get motif_pred=0 (no prediction) — no random fallback.
    normalization matches freq_loss (cosine distance = euclidean on unit sphere).
    """
    hash_table = LSHash(hash_dim, model.hidden_dim)
    hash_set   = set()
    results    = []

    for entry in cache:
        lvl        = entry['lvl']
        n_nodes    = entry['n_nodes']
        Z          = lvl['z_sub']       # (n_subs, hidden_dim)
        scores     = lvl['scores']      # (n_subs,)
        k_norm     = lvl['k_norm']      # (n_subs,) — normalized subgraph size ∈ [0,1]
        flat_nodes = lvl['flat_nodes']  # (n_pairs,)
        flat_subs  = lvl['flat_subs']   # (n_pairs,)

        if dummy:
            Z      = torch.randn_like(Z)
            scores = torch.rand_like(scores)

        k_max   = model.k_max
        k_vals  = (k_norm * max(k_max - 3, 1)).round().long() + 3
        bids    = torch.tensor([_k_to_bucket(k.item()) for k in k_vals])
        mask    = (bids == bucket)

        if mask.sum() == 0:
            # no subgraphs in this bucket for this graph — all nodes uncovered
            results.append({'hashes': None, 'scores': None, 'has_sub': None})
            continue

        # normalize before hashing — matches freq_loss knn density computation
        Z_norm = F.normalize(Z[mask], dim=-1)

        # map global sub indices → local bucket indices
        sub_remap = torch.full((Z.size(0),), -1, dtype=torch.long)
        sub_remap[mask] = torch.arange(mask.sum())

        flat_mask    = mask[flat_subs]
        flat_nodes_b = flat_nodes[flat_mask]
        flat_subs_b  = sub_remap[flat_subs[flat_mask]]
        scores_b     = scores[mask]

        sub_hashes = [hash_table.index(Z_norm[i].numpy())[0] for i in range(Z_norm.size(0))]
        hash_set.update(sub_hashes)

        # assign each node to its highest-scoring subgraph (ascending → last scatter wins)
        order     = torch.argsort(scores_b[flat_subs_b])
        node_best = torch.zeros(n_nodes, dtype=torch.long)
        node_best.scatter_(0, flat_nodes_b[order], flat_subs_b[order])

        has_sub = torch.zeros(n_nodes, dtype=torch.bool)
        has_sub[flat_nodes_b] = True

        node_scores = torch.zeros(n_nodes)
        node_hashes = [''] * n_nodes
        for v in has_sub.nonzero(as_tuple=True)[0].tolist():
            node_hashes[v] = sub_hashes[node_best[v].item()]
            node_scores[v] = scores_b[node_best[v]].item()

        results.append({'hashes': node_hashes, 'scores': node_scores, 'has_sub': has_sub})

    hash_idx = {h: i + 1 for i, h in enumerate(sorted(hash_set))}

    decoded = []
    for entry, r in zip(cache, results):
        if r['has_sub'] is None:
            # no subgraphs in this bucket — treat all nodes as uncovered (motif_pred=0)
            decoded.append({
                'motif_id':   entry['motif_id'],
                'cum_scores': torch.zeros(entry['n_nodes']),
                'motif_pred': torch.zeros(entry['n_nodes'], dtype=torch.long),
            })
            continue
        motif_pred = torch.zeros(entry['n_nodes'], dtype=torch.long)
        for v in r['has_sub'].nonzero(as_tuple=True)[0].tolist():
            motif_pred[v] = hash_idx[r['hashes'][v]]
        decoded.append({
            'motif_id':   entry['motif_id'],
            'cum_scores': torch.tensor(r['scores']),
            'motif_pred': motif_pred,
        })
    return decoded


def _eval_lsh(decoded_graphs, n_motifs=1, top_k=1):
    """faithful analog of HashDecoder.eval."""
    sigma_all       = torch.cat([g['cum_scores'] for g in decoded_graphs])
    motifs_pred_all = torch.cat([g['motif_pred'] for g in decoded_graphs])
    true_motif_ids  = torch.cat([g['motif_id']   for g in decoded_graphs])

    # reindex motif ids to contiguous 0..K
    motifs_input  = torch.unique(motifs_pred_all)
    motif_indices = {m.item(): i for i, m in enumerate(motifs_input)}
    for i in range(len(motifs_pred_all)):
        motifs_pred_all[i] = motif_indices[motifs_pred_all[i].item()]

    # rank hash buckets by average score
    motif_ids, counts = torch.unique(motifs_pred_all, return_counts=True)
    sigma_avg = torch.zeros_like(motif_ids, dtype=torch.float32).scatter_add(
        0, motifs_pred_all, sigma_all) / counts
    motifs_sorted = torch.argsort(sigma_avg, descending=True)
    ranks = torch.zeros_like(motifs_sorted)
    for ind, val in enumerate(motifs_sorted):
        ranks[val] = ind

    # keep top_k buckets; zero out the rest
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
                        help='fast sanity check: n_graphs=30, n_runs=1')
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

    model = MotiFiestaDisc(
        n_features = hp['n_features'],
        hidden_dim = hp.get('hidden_dim', 64),
        gin_layers = hp.get('gin_layers', 2),
        wl_hops    = hp.get('wl_hops', 0),
        k_max      = hp.get('k_max', 12),
        n_samples  = hp.get('n_samples', 50),
        pool       = hp.get('pool', 'mean'),
        dist_label = hp.get('dist_label', False),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    dataset   = get_loader(root=args.data, name=args.dataset)
    hash_dims = [8, 16, 32]

    print(f"\n=== {args.name} (top_k={args.top_k}, n_runs={args.n_runs}, "
          f"n_graphs={args.n_graphs}) ===\n")
    print("embedding graphs...", flush=True)
    cache = _embed_all(model, dataset, device, args.n_graphs)
    print(f"cached {len(cache)} graphs\n", flush=True)

    bucket_names = {0: 's (k≤4)', 1: 'm (k5-8)', 2: 'l (k9+)'}
    print(f"lsh jaccard (hash_dim∈{hash_dims} × bucket∈[s,m,l]):")
    best = {'mean': 0.0, 'std': 0.0, 'bucket': None, 'bucket_id': None, 'hash_dim': None}
    for hash_dim in hash_dims:
        for bucket in [0, 1, 2]:
            run_scores = []
            for _ in range(args.n_runs):
                dg = _decode(model, cache, hash_dim, bucket)
                run_scores.append(_eval_lsh(dg, n_motifs=1, top_k=args.top_k))
            mean_s = statistics.mean(run_scores)
            std_s  = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
            marker = ' *' if mean_s > best['mean'] else ''
            print(f"  hash_dim={hash_dim}  bucket={bucket_names[bucket]}:  {mean_s:.4f} ±{std_s:.4f}{marker}")
            if mean_s > best['mean']:
                best = {'mean': mean_s, 'std': std_s,
                        'bucket': bucket_names[bucket], 'bucket_id': bucket,
                        'hash_dim': hash_dim}

    dummy_scores = []
    for _ in range(args.n_runs):
        dg = _decode(model, cache, best['hash_dim'], best['bucket_id'], dummy=True)
        dummy_scores.append(_eval_lsh(dg, n_motifs=1, top_k=args.top_k))
    dm = statistics.mean(dummy_scores)
    ds = statistics.stdev(dummy_scores) if len(dummy_scores) > 1 else 0.0

    print(f"\nbest:  {best['mean']:.2f} ±{best['std']:.2f}  "
          f"(hash_dim={best['hash_dim']}, bucket={best['bucket']})")
    print(f"dummy: {dm:.2f} ±{ds:.2f}")
    print(f"\nfinal: {best['mean']:.2f} ±{best['std']:.2f}  ({dm:.2f} ±{ds:.2f})")


if __name__ == '__main__':
    main()
    import sys; sys.exit(0)
