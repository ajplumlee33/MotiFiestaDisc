"""rand-esu motif discovery: przulj isomorphism certificate + cross-graph frequency scoring.
each subgraph is typed by tuple(sorted((degree, triangle_count) per node)).
certificate tuple is the bucket key directly — no lsh or learned projection.
bucket score = (avg_freq_pos)^alpha / (avg_freq_pos + avg_freq_neg).
"""
import argparse, collections, sys, warnings
warnings.filterwarnings('ignore', category=UserWarning)

import networkx as nx

from MotiFiesta.training.loading import get_loader
from MotiFiesta.disc.eval import sample, build_adj, eval_jaccard


def _iso_certificate(nx_graph):
    """przulj isomorphism class certificate via sorted per-node (degree, triangle) pairs."""
    tri = nx.triangles(nx_graph)
    return tuple(sorted((d, tri[n]) for n, d in nx_graph.degree()))


def _to_nx(nodes, adj):
    nm = {v: i for i, v in enumerate(nodes)}
    G  = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for u in nodes:
        for v in adj[u]:
            if v in nm and nm[u] < nm[v]:
                G.add_edge(nm[u], nm[v])
    return G


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser()
    p.add_argument('--name',        required=True)
    p.add_argument('--data',        required=True)
    p.add_argument('--dataset',     default='synth_pairs')
    p.add_argument('--n-samples',   type=int, default=20)
    p.add_argument('--k-min',       type=int, default=3)
    p.add_argument('--k-max',       type=int, default=4)
    p.add_argument('--n-graphs',    type=int, default=0)
    p.add_argument('--score-alpha', type=float, default=1.5)
    p.add_argument('--top-k',       type=int, default=3)
    p.add_argument('--pred-mode',   default='bucket', choices=['bucket', 'subgraph'])
    p.add_argument('--cc-filter',   action='store_true')
    args = p.parse_args()

    dataset    = get_loader(root=args.data, name=args.dataset)
    all_graphs = list(dataset['dataset_whole'])
    if args.n_graphs > 0:
        all_graphs = all_graphs[:args.n_graphs]
    n_g    = len(all_graphs)
    split  = int(0.8 * n_g)
    train_graphs = all_graphs[:split]
    test_graphs  = all_graphs[split:]
    print(f"\n=== {args.name}  n_graphs={n_g}  train={len(train_graphs)}  test={len(test_graphs)}"
          f"  k=[{args.k_min},{args.k_max}] ===\n")

    # fitting pass: count certificate types across train graphs
    pos_canon_per_graph = []
    neg_canon_per_graph = []

    for idx, pair in enumerate(train_graphs):
        g_pos   = pair['pos']
        adj_pos = build_adj(g_pos.edge_index.cpu(), len(g_pos.x))
        pos_types = collections.Counter()
        for ns in sample(adj_pos, len(g_pos.x), args.n_samples, args.k_min, args.k_max):
            pos_types[_iso_certificate(_to_nx(ns, adj_pos))] += 1
        pos_canon_per_graph.append(pos_types)

        g_neg   = pair['neg']
        adj_neg = build_adj(g_neg.edge_index.cpu(), len(g_neg.x))
        neg_types = collections.Counter()
        for ns in sample(adj_neg, len(g_neg.x), args.n_samples, args.k_min, args.k_max):
            neg_types[_iso_certificate(_to_nx(ns, adj_neg))] += 1
        neg_canon_per_graph.append(neg_types)

        if (idx + 1) % 100 == 0:
            print(f"  train {idx+1}/{len(train_graphs)}", flush=True)

    # score certificate types by cross-graph average frequency ratio
    n_pos_g   = len(pos_canon_per_graph)
    n_neg_g   = len(neg_canon_per_graph)
    all_types = set(t for g in pos_canon_per_graph for t in g)
    r_pos = collections.Counter({t: sum(g[t] for g in pos_canon_per_graph) for t in all_types})
    r_neg = collections.Counter({t: sum(g[t] for g in neg_canon_per_graph) for t in all_types})
    alpha = args.score_alpha
    bucket_score = {
        t: (r_pos[t] / n_pos_g) ** alpha / ((r_pos[t] / n_pos_g) + (r_neg[t] / n_neg_g) + 1e-8)
        for t in r_pos
    }
    print(f"  isomorphism classes (przulj): {len(all_types)} distinct types")

    # evaluation pass: embed test graphs and score
    test_subs = []
    true_sets = []
    adjs      = []
    for pair in test_graphs:
        g_pos   = pair['pos']
        adj_pos = build_adj(g_pos.edge_index.cpu(), len(g_pos.x))
        tru     = set(g_pos.motif_id.nonzero(as_tuple=False).squeeze(-1).tolist())
        subs = [
            (ns, _iso_certificate(_to_nx(ns, adj_pos)))
            for ns in sample(adj_pos, len(g_pos.x), args.n_samples, args.k_min, args.k_max)
        ]
        test_subs.append(subs)
        true_sets.append(tru)
        adjs.append(adj_pos)

    graph_subs = [[(ns, z) for ns, z in subs] for subs in test_subs]
    eval_jaccard(graph_subs, true_sets, adjs, bucket_score, r_pos,
                 args.top_k, args.pred_mode, args.cc_filter)


if __name__ == '__main__':
    main()
    sys.exit(0)
