"""
graph classification on tudatasets using rand-esu + wl + lda simhash.
per-graph feature: bucket count histogram over simhash assignments of sampled subgraphs.
lda planes are fit per fold from class 0 (neg) vs class 1 (pos) subgraph clouds
on training graphs only. 10-fold stratified cv with random forest, matching
disc_classify.py protocol.

run from repo root:
    python MotiFiesta/disc/lda_classify.py --dataset PROTEINS
"""
import argparse, collections, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from MotiFiesta.disc.decode import (
    _build_adj, _to_nx, _wl_embed, _simhash, _fit_planes, _sample,
)
from MotiFiesta.utils.real_world import TUDataset


def _graph_zs(g, args):
    adj        = _build_adj(g.edge_index.cpu(), g.num_nodes)
    subs       = _sample(adj, g.num_nodes, args.n_samples, args.k_min, args.k_max)
    node_feats = g.x.numpy() if g.x is not None else None
    return [_wl_embed(_to_nx(ns, adj), ns, dim=args.embed_dim, node_feats=node_feats)
            for ns in subs]


def _fit_fold_planes(train_zs, train_ys, args):
    pos_zs = [z for zs, y in zip(train_zs, train_ys) for z in zs if y == 1]
    neg_zs = [z for zs, y in zip(train_zs, train_ys) for z in zs if y == 0]
    Z_pos  = np.array(pos_zs, dtype=np.float32) if pos_zs else np.zeros((1, args.embed_dim))
    Z_neg  = np.array(neg_zs, dtype=np.float32) if neg_zs else np.zeros((1, args.embed_dim))
    if args.proj == 'lda':
        planes, sep = _fit_planes(Z_pos, Z_neg, args.hash_dim, args.embed_dim)
        return planes, f"lda sep={sep:.3f}"
    else:
        Z_all  = np.vstack([Z_pos, Z_neg])
        n_comp = min(args.hash_dim, Z_all.shape[0] - 1, args.embed_dim - 1)
        pca    = PCA(n_components=n_comp, random_state=42)
        pca.fit(Z_all)
        return pca.components_, f"pca var={pca.explained_variance_ratio_.sum():.3f}"


def _graph_feat(zs, planes, vocab):
    # mean-pooled wl embedding (dense, 32-dim analog of motifiesta's global_add_pool z_sub)
    wl_pool = np.mean(zs, axis=0) if zs else np.zeros(zs[0].shape if zs else (32,))
    # bucket count histogram (sparse, vocab-dim)
    hist = np.zeros(len(vocab), dtype=np.float32)
    for z in zs:
        h = _simhash(z, planes)
        if h in vocab:
            hist[vocab[h]] += 1.0
    return np.concatenate([wl_pool, hist])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',      default='PROTEINS')
    ap.add_argument('--n-samples',    type=int,   default=50)
    ap.add_argument('--k-min',        type=int,   default=3)
    ap.add_argument('--k-max',        type=int,   default=4)
    ap.add_argument('--embed-dim',    type=int,   default=32)
    ap.add_argument('--hash-dim',     type=int,   default=16)
    ap.add_argument('--proj',         default='lda', choices=['lda', 'pca'])
    ap.add_argument('--n-folds',      type=int,   default=10)
    ap.add_argument('--n-estimators', type=int,   default=500)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    dataset = TUDataset(root='data', name=args.dataset)
    ys      = np.array([dataset[i].y.item() for i in range(len(dataset))])

    print(f"dataset: {args.dataset}  n={len(dataset)}  classes={sorted(set(ys.tolist()))}")
    print(f"proj={args.proj}  hash_dim={args.hash_dim}  k=[{args.k_min},{args.k_max}]"
          f"  n_samples={args.n_samples}\n")
    print("sampling subgraphs...")

    all_zs = []
    for i in range(len(dataset)):
        all_zs.append(_graph_zs(dataset[i], args))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dataset)}", flush=True)

    cv      = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    rf_accs = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.arange(len(dataset)), ys)):
        train_zs = [all_zs[i] for i in train_idx]
        test_zs  = [all_zs[i] for i in test_idx]
        train_ys = ys[train_idx]
        test_ys  = ys[test_idx]

        planes, proj_info = _fit_fold_planes(train_zs, train_ys, args)

        vocab = {}
        for zs in train_zs:
            for z in zs:
                h = _simhash(z, planes)
                if h not in vocab:
                    vocab[h] = len(vocab)

        X_train = np.array([_graph_feat(zs, planes, vocab) for zs in train_zs])
        X_test  = np.array([_graph_feat(zs, planes, vocab) for zs in test_zs])

        clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                     random_state=42, n_jobs=-1)
        clf.fit(X_train, train_ys)
        rf_acc = (clf.predict(X_test) == test_ys).mean()
        rf_accs.append(rf_acc)

        print(f"fold {fold+1:2d}/{args.n_folds}:  acc={rf_acc:.4f}"
              f"  vocab={len(vocab)}  {proj_info}")

    print(f"\n{args.dataset}  {args.n_folds}-fold acc:  {np.mean(rf_accs):.4f} ± {np.std(rf_accs):.4f}")
    print(f"motifiesta:  0.7310 ± 0.0200")


if __name__ == '__main__':
    main()
