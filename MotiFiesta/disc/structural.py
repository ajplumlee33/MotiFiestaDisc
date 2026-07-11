"""rand-esu motif discovery: wl / wwl / spectral / tree / canonical embedding modes."""
import argparse, sys, warnings
warnings.filterwarnings('ignore', category=UserWarning)

import networkx as nx
import numpy as np

from MotiFiesta.training.loading import get_loader
from MotiFiesta.disc.eval import (
    sample, build_adj, simhash, fit, save_model, load_model, eval_jaccard
)


# ─── embedding functions ──────────────────────────────────────────────────────

def _wl_embed(nx_graph, nodes, num_iter=4, dim=32, node_feats=None):
    """wl label propagation histogram via hash trick."""
    if node_feats is not None:
        if ((node_feats == 0) | (node_feats == 1)).all() and (node_feats.sum(axis=1) == 1).all():
            labels = {i: str(int(node_feats[nodes[i]].argmax())) for i in nx_graph.nodes()}
        else:
            labels = {i: str(node_feats[nodes[i]].tolist()) for i in nx_graph.nodes()}
    else:
        labels = {i: str(nx_graph.degree(i)) for i in nx_graph.nodes()}
    vec = np.zeros(dim, dtype=np.float32)
    for label in labels.values():
        vec[abs(hash(label)) % dim] += 1
    for _ in range(num_iter):
        new_labels = {}
        for n in nx_graph.nodes():
            nbr = sorted(labels[u] for u in nx_graph.neighbors(n))
            new_labels[n] = str(hash((labels[n], tuple(nbr))))
        labels = new_labels
        for label in labels.values():
            vec[abs(hash(label)) % dim] += 1
    return vec


_LABEL_VEC_CACHE: dict = {}

def _label_vec(label, dim):
    key = (label, dim)
    if key not in _LABEL_VEC_CACHE:
        rng = np.random.RandomState(abs(hash(label)) % (2**31))
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        _LABEL_VEC_CACHE[key] = v
    return _LABEL_VEC_CACHE[key]


def _wwl_embed(nx_graph, nodes, num_iter=4, dim=32, node_feats=None):
    """mean-pooled per-node wl label vectors across iterations."""
    if node_feats is not None:
        if ((node_feats == 0) | (node_feats == 1)).all() and (node_feats.sum(axis=1) == 1).all():
            labels = {i: str(int(node_feats[nodes[i]].argmax())) for i in nx_graph.nodes()}
        else:
            labels = {i: str(node_feats[nodes[i]].tolist()) for i in nx_graph.nodes()}
    else:
        labels = {i: str(nx_graph.degree(i)) for i in nx_graph.nodes()}
    iter_dim = max(1, dim // (num_iter + 1))
    pools = []
    for _ in range(num_iter + 1):
        pools.append(np.mean([_label_vec(labels[n], iter_dim) for n in nx_graph.nodes()], axis=0))
        new_labels = {}
        for n in nx_graph.nodes():
            nbr = sorted(labels[u] for u in nx_graph.neighbors(n))
            new_labels[n] = str(hash((labels[n], tuple(nbr))))
        labels = new_labels
    vec = np.concatenate(pools).astype(np.float32)
    if len(vec) < dim:
        vec = np.pad(vec, (0, dim - len(vec)))
    return vec[:dim]


def _spectral_embed(nx_graph, nodes, dim=32, node_feats=None, **kwargs):
    """sorted adjacency eigenvalues zero-padded to dim."""
    A = nx.to_numpy_array(nx_graph)
    eigs = np.sort(np.linalg.eigvalsh(A))[::-1].astype(np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    vec[:min(len(eigs), dim)] = eigs[:dim]
    return vec


def _tree_features(nx_graph, nodes=None, dim=None, node_feats=None, **kwargs):
    """structural feature vector: size, min degree, triangles, density, sorted degree sequence."""
    n = nx_graph.number_of_nodes()
    m = nx_graph.number_of_edges()
    degs = sorted([d for _, d in nx_graph.degree()], reverse=True)
    deg_pad = np.zeros(16, dtype=np.float32)
    deg_pad[:len(degs)] = degs[:16]
    triangles = float(sum(nx.triangles(nx_graph).values()) // 3)
    density = 2.0 * m / max(n * (n - 1), 1)
    return np.concatenate([[n, float(min(degs) if degs else 0), triangles, density],
                           deg_pad]).astype(np.float32)


def _to_nx(nodes, adj):
    nm = {v: i for i, v in enumerate(nodes)}
    G  = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for u in nodes:
        for v in adj[u]:
            if v in nm and nm[u] < nm[v]:
                G.add_edge(nm[u], nm[v])
    return G


# ─── main ─────────────────────────────────────────────────────────────────────

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
    p.add_argument('--embed-mode',  default='wl', choices=['wl', 'wwl', 'spectral', 'tree'])
    p.add_argument('--score-alpha', type=float, default=1.5)
    p.add_argument('--clf',         default='auto', choices=['auto', 'lda', 'tree'])
    p.add_argument('--tree-depth',  type=int, default=5)
    p.add_argument('--top-k',       type=int, default=3)
    p.add_argument('--pred-mode',   default='bucket', choices=['bucket', 'subgraph'])
    p.add_argument('--proj',        default='lda', choices=['lda', 'pca'])
    p.add_argument('--score-mode',  default='gstat', choices=['count', 'knn', 'gstat'])
    p.add_argument('--knn-k',       type=int, default=30)
    p.add_argument('--cc-filter',   action='store_true')
    p.add_argument('--save-model',  default=None, metavar='PATH')
    p.add_argument('--load-model',  default=None, metavar='PATH')
    args = p.parse_args()

    dataset    = get_loader(root=args.data, name=args.dataset)
    all_graphs = list(dataset['dataset_whole'])
    if args.n_graphs > 0:
        all_graphs = all_graphs[:args.n_graphs]
    n_g = len(all_graphs)
    split = int(0.8 * n_g)
    train_graphs = all_graphs[:split]
    test_graphs  = all_graphs[split:]
    print(f"\n=== {args.name}  n_graphs={n_g}  train={len(train_graphs)}  test={len(test_graphs)}"
          f"  k=[{args.k_min},{args.k_max}]  hash_dim={args.hash_dim}  embed_dim={args.embed_dim} ===\n")

    embed_fn = {'wwl': _wwl_embed, 'spectral': _spectral_embed,
                'tree': _tree_features}.get(args.embed_mode, _wl_embed)

    # fitting pass: collect embeddings from train graphs (pos + neg)
    Z_pos_list  = []
    Z_neg_list  = []
    Z_pos_sizes = []
    Z_neg_sizes = []

    if not args.load_model:
        for idx, pair in enumerate(train_graphs):
            g_pos   = pair['pos']
            adj_pos = build_adj(g_pos.edge_index.cpu(), len(g_pos.x))
            x_pos   = g_pos.x.numpy()
            for ns in sample(adj_pos, len(g_pos.x), args.n_samples, args.k_min, args.k_max):
                z = embed_fn(_to_nx(ns, adj_pos), ns, dim=args.embed_dim, node_feats=x_pos)
                Z_pos_list.append(z); Z_pos_sizes.append(len(ns))

            g_neg   = pair['neg']
            adj_neg = build_adj(g_neg.edge_index.cpu(), len(g_neg.x))
            x_neg   = g_neg.x.numpy()
            for ns in sample(adj_neg, len(g_neg.x), args.n_samples, args.k_min, args.k_max):
                z = embed_fn(_to_nx(ns, adj_neg), ns, dim=args.embed_dim, node_feats=x_neg)
                Z_neg_list.append(z); Z_neg_sizes.append(len(ns))

            if (idx + 1) % 100 == 0:
                print(f"  train {idx+1}/{len(train_graphs)}", flush=True)

        if Z_pos_list:
            args.embed_dim = np.array(Z_pos_list).shape[1]

    # evaluation pass: collect embeddings from test graphs (pos only)
    test_subs = []
    true_sets = []
    adjs      = []
    for idx, pair in enumerate(test_graphs):
        g_pos   = pair['pos']
        adj_pos = build_adj(g_pos.edge_index.cpu(), len(g_pos.x))
        tru     = set(g_pos.motif_id.nonzero(as_tuple=False).squeeze(-1).tolist())
        x_pos   = g_pos.x.numpy()
        subs = [
            (ns, embed_fn(_to_nx(ns, adj_pos), ns, dim=args.embed_dim, node_feats=x_pos))
            for ns in sample(adj_pos, len(g_pos.x), args.n_samples, args.k_min, args.k_max)
        ]
        test_subs.append(subs)
        true_sets.append(tru)
        adjs.append(adj_pos)

        if (idx + 1) % 100 == 0:
            print(f"  test {idx+1}/{len(test_graphs)}", flush=True)

    if args.embed_mode != 'canonical' and Z_pos_list:
        args.embed_dim = np.array(Z_pos_list).shape[1]

    # fit or load
    if args.load_model:
        planes, tree_clf, r_pos, r_neg, bucket_score = load_model(args.load_model)
    else:
        planes, tree_clf, r_pos, r_neg, bucket_score = fit(
            Z_pos_list, Z_neg_list, Z_pos_sizes, Z_neg_sizes,
            [], [], args
        )
        if args.save_model:
            save_model(args.save_model, planes, tree_clf, bucket_score, r_pos, r_neg)

    # assign test subgraphs to buckets and evaluate
    def _bucket(z):
        if tree_clf is not None:
            return int(tree_clf.apply(z.reshape(1, -1))[0])
        return simhash(z, planes)

    graph_subs = [[(ns, _bucket(z)) for ns, z in subs] for subs in test_subs]

    eval_jaccard(graph_subs, true_sets, adjs, bucket_score, r_pos,
                 args.top_k, args.pred_mode, args.cc_filter)


if __name__ == '__main__':
    main()
    sys.exit(0)
