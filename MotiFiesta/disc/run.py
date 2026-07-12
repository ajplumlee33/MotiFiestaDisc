"""rand-esu motif discovery experiment runner."""
import argparse, collections, sys, warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np

from MotiFiesta.training.loading import get_loader
from MotiFiesta.disc.embed import EMBED_FNS, to_nx, canonical_cert
from MotiFiesta.disc.eval import (
    sample, build_adj, simhash,
    fit, save_model, load_model, eval_jaccard,
)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser()
    p.add_argument('--name',        required=True)
    p.add_argument('--data',        required=True)
    p.add_argument('--dataset',     default='synth_pairs')
    p.add_argument('--n-samples',   type=int, default=20)
    p.add_argument('--k-min',       type=int, default=3)
    p.add_argument('--k-max',       type=int, default=4)
    p.add_argument('--hash-dim',    type=int, default=16)
    p.add_argument('--embed-dim',   type=int, default=32)
    p.add_argument('--n-graphs',    type=int, default=0)
    p.add_argument('--embed-mode',  default='wl',
                   choices=['wl', 'wwl', 'tree', 'canonical'])
    p.add_argument('--clf',         default='auto',
                   choices=['auto', 'lda', 'tree', 'knn'])
    p.add_argument('--score',       default='gstat', choices=['gstat', 'count'])
    p.add_argument('--tree-depth',  type=int, default=5)
    p.add_argument('--score-alpha', type=float, default=1.5)
    p.add_argument('--knn-k',       type=int, default=30)
    p.add_argument('--top-k',       type=int, default=3)
    p.add_argument('--pred-mode',   default='bucket', choices=['bucket', 'subgraph'])
    p.add_argument('--cc-filter',   action='store_true')
    p.add_argument('--save-model',  default=None, metavar='PATH')
    p.add_argument('--load-model',  default=None, metavar='PATH')
    args = p.parse_args()

    if args.clf == 'auto':
        args.clf = 'tree' if args.embed_mode == 'tree' else 'lda'

    embed_fn = EMBED_FNS[args.embed_mode]

    dataset    = get_loader(root=args.data, name=args.dataset)
    all_graphs = list(dataset['dataset_whole'])
    if args.n_graphs > 0:
        all_graphs = all_graphs[:args.n_graphs]
    n_g    = len(all_graphs)
    split  = int(0.8 * n_g)
    train_graphs = all_graphs[:split]
    test_graphs  = all_graphs[split:]
    print(f"\n=== {args.name}  embed={args.embed_mode}  clf={args.clf}"
          f"  n_graphs={n_g}  train={len(train_graphs)}  test={len(test_graphs)}"
          f"  k=[{args.k_min},{args.k_max}] ===\n")

    Z_pos_list  = []; Z_neg_list  = []
    Z_pos_sizes = []; Z_neg_sizes = []
    pos_canon_per_graph = []; neg_canon_per_graph = []

    if not args.load_model:
        for idx, pair in enumerate(train_graphs):
            g_pos   = pair['pos']
            adj_pos = build_adj(g_pos.edge_index.cpu(), len(g_pos.x))
            x_pos   = g_pos.x.numpy()
            g_neg   = pair['neg']
            adj_neg = build_adj(g_neg.edge_index.cpu(), len(g_neg.x))
            x_neg   = g_neg.x.numpy()

            if args.embed_mode == 'canonical':
                pos_counter = collections.Counter()
                for ns in sample(adj_pos, len(g_pos.x), args.n_samples, args.k_min, args.k_max):
                    pos_counter[canonical_cert(to_nx(ns, adj_pos))] += 1
                pos_canon_per_graph.append(pos_counter)

                neg_counter = collections.Counter()
                for ns in sample(adj_neg, len(g_neg.x), args.n_samples, args.k_min, args.k_max):
                    neg_counter[canonical_cert(to_nx(ns, adj_neg))] += 1
                neg_canon_per_graph.append(neg_counter)
            else:
                for ns in sample(adj_pos, len(g_pos.x), args.n_samples, args.k_min, args.k_max):
                    z = embed_fn(to_nx(ns, adj_pos), ns, dim=args.embed_dim, node_feats=x_pos)
                    Z_pos_list.append(z); Z_pos_sizes.append(len(ns))
                for ns in sample(adj_neg, len(g_neg.x), args.n_samples, args.k_min, args.k_max):
                    z = embed_fn(to_nx(ns, adj_neg), ns, dim=args.embed_dim, node_feats=x_neg)
                    Z_neg_list.append(z); Z_neg_sizes.append(len(ns))

            if (idx + 1) % 100 == 0:
                print(f"  train {idx+1}/{len(train_graphs)}", flush=True)

        if Z_pos_list:
            args.embed_dim = np.array(Z_pos_list).shape[1]

    test_subs = []; true_sets = []; adjs = []
    for idx, pair in enumerate(test_graphs):
        g_pos   = pair['pos']
        adj_pos = build_adj(g_pos.edge_index.cpu(), len(g_pos.x))
        tru     = set(g_pos.motif_id.nonzero(as_tuple=False).squeeze(-1).tolist())
        x_pos   = g_pos.x.numpy()

        if args.embed_mode == 'canonical':
            subs = [(ns, canonical_cert(to_nx(ns, adj_pos)))
                    for ns in sample(adj_pos, len(g_pos.x), args.n_samples, args.k_min, args.k_max)]
        else:
            subs = [(ns, embed_fn(to_nx(ns, adj_pos), ns, dim=args.embed_dim, node_feats=x_pos))
                    for ns in sample(adj_pos, len(g_pos.x), args.n_samples, args.k_min, args.k_max)]

        test_subs.append(subs); true_sets.append(tru); adjs.append(adj_pos)

        if (idx + 1) % 100 == 0:
            print(f"  test {idx+1}/{len(test_graphs)}", flush=True)

    if args.load_model:
        planes, tree_clf, r_pos, r_neg, bucket_score = load_model(args.load_model)
    else:
        planes, tree_clf, r_pos, r_neg, bucket_score = fit(
            Z_pos_list, Z_neg_list, Z_pos_sizes, Z_neg_sizes,
            pos_canon_per_graph, neg_canon_per_graph, args
        )
        if args.save_model:
            save_model(args.save_model, planes, tree_clf, bucket_score, r_pos, r_neg)

    def _bucket(z):
        if args.embed_mode == 'canonical':
            return z
        if tree_clf is not None:
            return int(tree_clf.apply(z.reshape(1, -1))[0])
        return simhash(z, planes)

    graph_subs = [[(ns, _bucket(z)) for ns, z in subs] for subs in test_subs]
    eval_jaccard(graph_subs, true_sets, adjs, bucket_score, r_pos,
                 args.top_k, args.pred_mode, args.cc_filter)


if __name__ == '__main__':
    main()
    sys.exit(0)
