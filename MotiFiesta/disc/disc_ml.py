"""
evaluate rand-esu + wl feature hashing + lda-pca simhash for motif discovery.

projection planes are fit in two stages using pos and neg subgraph embeddings:
  plane 0:   lda discriminant direction w = S_W^{-1}(mu_pos - mu_neg), the direction
             in wl space that maximally separates pos from neg subgraph clouds.
  planes 1+: pca components of the lda-residual pos embedding space.

bucket scores use frequency x purity:
  score(h) = r_pos(h)^alpha / (r_pos(h) + r_neg(h))
neg graphs are used only in the fitting phase; inference uses stored scores only.
"""
import argparse, collections, pickle, random, sys, warnings
warnings.filterwarnings('ignore', category=UserWarning)

import networkx as nx
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from torch_geometric.utils import remove_self_loops, coalesce

from MotiFiesta.training.loading import get_loader
from MotiFiesta.disc.disc_model import build_subgraph_tensors, load_gae


# ─── rand-esu ─────────────────────────────────────────────────────────────────

_ENUM_BUDGET = 50

def _est_p_ext(n_g, avg_deg):
    return min(1.0, (_ENUM_BUDGET / max(1.0, n_g * max(avg_deg, 1.0)**2 / 3))**0.5)

def _esu_extend(sub, ext, min_v, ns, adj, p, out, k_min, k_max, budget):
    l = len(sub)
    if l >= 3:
        out.setdefault(l, []).append(list(sub))
        if l >= k_min:
            budget[0] -= 1
    if l == k_max or budget[0] <= 0:
        return
    for i, w in enumerate(ext):
        if budget[0] <= 0:
            return
        if random.random() > p:
            continue
        excl = sub | set(ext[:i+1])
        nxt  = list(ext[i+1:])
        seen = set(nxt) | excl
        for u in adj[w]:
            if u in ns and u > min_v and u not in seen:
                nxt.append(u); seen.add(u)
        _esu_extend(sub|{w}, nxt, min_v, ns, adj, p, out, k_min, k_max, budget)

def _rand_esu(nodes, ns, adj, p, k_min, k_max, budget):
    out = {}
    for v in nodes:
        if budget[0] <= 0:
            break
        ext = [u for u in adj[v] if u in ns and u > v]
        _esu_extend({v}, ext, v, ns, adj, p, out, k_min, k_max, budget)
    return out

def _sample(adj, n_nodes, n_samples, k_min, k_max):
    nodes = list(range(n_nodes))
    ns    = set(nodes)
    k_max = min(k_max, n_nodes - 1)
    if k_max < k_min:
        return []
    avg_d = sum(len(adj[v]) for v in nodes) / max(n_nodes, 1)
    p     = _est_p_ext(n_nodes, avg_d)
    by_k  = {}
    budget = [_ENUM_BUDGET * 100]
    for _ in range(5):
        budget[0] = _ENUM_BUDGET * 100
        for l, subs in _rand_esu(nodes, ns, adj, p, k_min, k_max, budget).items():
            if l >= k_min:
                by_k.setdefault(l, []).extend(subs)
        if by_k and all(len(v) >= _ENUM_BUDGET for v in by_k.values()):
            break
    all_subs = [s for subs in by_k.values() for s in subs]
    if not all_subs:
        return []
    return (random.choices(all_subs, k=n_samples)
            if len(all_subs) < n_samples else random.sample(all_subs, n_samples))


# ─── graph utils ──────────────────────────────────────────────────────────────

def _build_adj(ei, n):
    ei, _ = remove_self_loops(ei)
    ei, _ = coalesce(ei, None, num_nodes=n)
    a = [[] for _ in range(n)]
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        a[u].append(v)
    return a

def _to_nx(nodes, adj):
    nm = {v: i for i, v in enumerate(nodes)}
    G  = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for u in nodes:
        for v in adj[u]:
            if v in nm and nm[u] < nm[v]:
                G.add_edge(nm[u], nm[v])
    return G


# ─── wl feature embedding ─────────────────────────────────────────────────────

def _wl_embed(nx_graph, nodes, num_iter=4, dim=32, node_feats=None):
    """wl label propagation, raw label count histogram via hash trick."""
    if node_feats is not None:
        # one-hot features (e.g. one-hot degree) -> use class index for compact, collision-free labels
        # general features -> use full vector string
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


def _iso_certificate(nx_graph, nodes=None, **kwargs):
    """isomorphism class certificate via sorted per-node (degree, triangle) pairs.
    two subgraphs share a certificate iff they have the same node orbit signature,
    which is equivalent to true isomorphism for k<=5 and a coarse approximation for larger k."""
    tri = nx.triangles(nx_graph)
    return tuple(sorted((d, tri[n]) for n, d in nx_graph.degree()))


_LABEL_VEC_CACHE: dict = {}

def _label_vec(label, dim):
    key = (label, dim)
    if key not in _LABEL_VEC_CACHE:
        rng = np.random.RandomState(abs(hash(label)) % (2**31))
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        _LABEL_VEC_CACHE[key] = v
    return _LABEL_VEC_CACHE[key]


def _tree_features(nx_graph, nodes=None, dim=None, node_feats=None, **kwargs):
    """structural feature vector: size, min degree, triangles, density, sorted degree sequence.
    max_deg and edge count omitted (redundant with deg_pad[0] and density+n respectively)."""
    n = nx_graph.number_of_nodes()
    m = nx_graph.number_of_edges()
    degs = sorted([d for _, d in nx_graph.degree()], reverse=True)
    deg_pad = np.zeros(16, dtype=np.float32)
    deg_pad[:len(degs)] = degs[:16]
    triangles = float(sum(nx.triangles(nx_graph).values()) // 3)
    density = 2.0 * m / max(n * (n - 1), 1)
    return np.concatenate([[n, float(min(degs) if degs else 0), triangles, density],
                           deg_pad]).astype(np.float32)


def _spectral_embed(nx_graph, nodes, dim=32, node_feats=None, **kwargs):
    """sorted adjacency eigenvalues zero-padded to dim.
    invariant to node ordering, more expressive than 1-wl for many graph pairs."""
    A = nx.to_numpy_array(nx_graph)
    eigs = np.sort(np.linalg.eigvalsh(A))[::-1].astype(np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    vec[:min(len(eigs), dim)] = eigs[:dim]
    return vec


def _wwl_embed(nx_graph, nodes, num_iter=4, dim=32, node_feats=None):
    """wwl embedding: per-node continuous label vectors mean-pooled across nodes.
    each node gets a dim-dimensional random unit vector per wl iteration, derived
    deterministically from its wl label. pooling across nodes and projecting gives
    a fixed-size embedding that preserves more per-node structural information than
    the hash-trick histogram."""
    if node_feats is not None:
        if ((node_feats == 0) | (node_feats == 1)).all() and (node_feats.sum(axis=1) == 1).all():
            labels = {i: str(int(node_feats[nodes[i]].argmax())) for i in nx_graph.nodes()}
        else:
            labels = {i: str(node_feats[nodes[i]].tolist()) for i in nx_graph.nodes()}
    else:
        labels = {i: str(nx_graph.degree(i)) for i in nx_graph.nodes()}

    iter_dim = max(1, dim // (num_iter + 1))
    per_iter_pools = []
    for _ in range(num_iter + 1):
        pool = np.mean([_label_vec(labels[n], iter_dim) for n in nx_graph.nodes()], axis=0)
        per_iter_pools.append(pool)
        new_labels = {}
        for n in nx_graph.nodes():
            nbr = sorted(labels[u] for u in nx_graph.neighbors(n))
            new_labels[n] = str(hash((labels[n], tuple(nbr))))
        labels = new_labels

    vec = np.concatenate(per_iter_pools).astype(np.float32)
    if len(vec) < dim:
        vec = np.pad(vec, (0, dim - len(vec)))
    return vec[:dim]



def _simhash(z, planes):
    """sign of projection onto each plane gives one bit per component."""
    return tuple((planes @ z > 0).astype(np.int8).tolist())


def _largest_component(nodes, adj):
    """largest connected component of the induced subgraph on a node set."""
    nodes   = set(nodes)
    visited = set()
    best    = set()
    for start in nodes:
        if start in visited:
            continue
        comp  = set()
        stack = [start]
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            visited.add(v)
            for u in adj[v]:
                if u in nodes and u not in comp:
                    stack.append(u)
        if len(comp) > len(best):
            best = comp
    return best


def _fit_planes(Z_pos, Z_neg, hash_dim, embed_dim):
    """
    fit lda discriminant direction from pos/neg clouds, then pca on the
    lda-residual pos embeddings for the remaining projection planes.
    """
    mu_pos = Z_pos.mean(axis=0)
    mu_neg = Z_neg.mean(axis=0)
    S_W = ((Z_pos - mu_pos).T @ (Z_pos - mu_pos) +
           (Z_neg - mu_neg).T @ (Z_neg - mu_neg))
    S_W += 1e-6 * np.eye(embed_dim)
    w_lda = np.linalg.solve(S_W, mu_pos - mu_neg).astype(np.float32)
    w_lda /= np.linalg.norm(w_lda)

    proj_pos = Z_pos @ w_lda
    proj_neg = Z_neg @ w_lda
    separation = (proj_pos.mean() - proj_neg.mean()) / (proj_pos.std() + proj_neg.std() + 1e-8)

    n_pca = min(hash_dim - 1, Z_pos.shape[0] - 1, embed_dim - 1)
    planes = w_lda[None, :]
    if n_pca > 0:
        Z_res = Z_pos - (Z_pos @ w_lda[:, None]) * w_lda[None, :]
        pca = PCA(n_components=n_pca, random_state=42)
        pca.fit(Z_res)
        planes = np.vstack([planes, pca.components_])

    return planes, separation


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser()
    p.add_argument('--name',       required=True)
    p.add_argument('--data',       required=True)
    p.add_argument('--dataset',    default='synth_pairs')
    p.add_argument('--n-samples',  type=int, default=20)
    p.add_argument('--k-min',      type=int, default=3)
    p.add_argument('--k-max',      type=int, default=4)
    p.add_argument('--hash-dim',   type=int, default=16,
                   help='total projection planes: 1 lda + (hash_dim-1) pca')
    p.add_argument('--embed-dim',  type=int, default=32)
    p.add_argument('--n-graphs',   type=int, default=0)
    p.add_argument('--embed-mode', default='wl', choices=['wl', 'wwl', 'gae', 'spectral', 'tree', 'canonical'],
                   help='wl: hash-trick histogram; wwl: mean-pooled per-node label vectors;'
                        ' gae: subgraph gcn encoder trained with edge prediction (requires --gae-model);'
                        ' spectral: sorted adjacency eigenvalues zero-padded to embed-dim;'
                        ' tree: decision tree on structural features, leaf ids as buckets (bypasses lsh)')
    p.add_argument('--score-alpha', type=float, default=1.5,
                   help='exponent on r_pos in base score: r_pos^alpha / (r_pos + r_neg)')
    p.add_argument('--clf', default='auto', choices=['auto', 'lda', 'tree'],
                   help='auto: tree when embed-mode=tree, lda otherwise; tree: decision tree on any embedding')
    p.add_argument('--tree-depth', type=int, default=5,
                   help='max depth of decision tree (clf=tree or embed-mode=tree)')
    p.add_argument('--gae-model',  default=None, metavar='PATH',
                   help='path to trained SubgraphGAE checkpoint (required for --embed-mode gae)')
    p.add_argument('--top-k',      type=int, default=3,
                   help='bucket mode: candidate buckets to try; subgraph mode: subgraphs per graph')
    p.add_argument('--pred-mode',  default='bucket', choices=['bucket', 'subgraph'],
                   help='bucket: union of best-scoring bucket; subgraph: top-k subgraphs ranked by (score, k)')
    p.add_argument('--proj',       default='lda', choices=['lda', 'pca'],
                   help='lda: 1 lda plane + (hash_dim-1) pca; pca: pure pca planes')
    p.add_argument('--score-mode',  default='gstat', choices=['count', 'knn', 'gstat'],
                   help='count: r_pos^alpha/(r_pos+r_neg) per bucket;'
                        ' knn: mean(d_neg/d_pos) per bucket, analogous to motifiesta sigma')
    p.add_argument('--knn-k',      type=int, default=30,
                   help='k for knn density ratio scoring (score-mode=knn)')
    p.add_argument('--cc-filter',  action='store_true',
                   help='restrict prediction to the largest connected component'
                        ' of the predicted node set in the pos graph')
    p.add_argument('--save-model', default=None, metavar='PATH',
                   help='after fitting, serialize planes and bucket scores to this path')
    p.add_argument('--load-model', default=None, metavar='PATH',
                   help='load planes and bucket scores from this path; skip neg sampling and fitting')
    args = p.parse_args()

    dataset    = get_loader(root=args.data, name=args.dataset)
    all_graphs = list(dataset['dataset_whole'])
    if args.n_graphs > 0:
        all_graphs = all_graphs[:args.n_graphs]
    n_g = len(all_graphs)

    print(f"\n=== {args.name}  n_graphs={n_g}  k=[{args.k_min},{args.k_max}]"
          f"  hash_dim={args.hash_dim}  embed_dim={args.embed_dim} ===\n")

    # embedding collection pass: sample from both pos and neg graphs
    raw_graph_subs = []  # list of (true_set, adj_pos, [(ns, z), ...]) for pos graphs
    Z_pos_list  = []
    Z_neg_list  = []
    Z_pos_sizes = []  # k-size of each pos subgraph (for knn scoring)
    Z_neg_sizes = []  # k-size of each neg subgraph

    gae_model = None
    if args.embed_mode == 'gae':
        if not args.gae_model:
            raise ValueError('--gae-model required with --embed-mode gae')
        gae_model, gae_out_dim = load_gae(args.gae_model)
        args.embed_dim = gae_out_dim

    embed_fn = {'wwl': _wwl_embed, 'spectral': _spectral_embed,
                'tree': _tree_features}.get(args.embed_mode, _wl_embed)

    neg_canon_per_graph = []  # list of sets, one per neg graph (canonical mode only)

    for idx, pair in enumerate(all_graphs):
        g_pos   = pair['pos']
        n_pos   = len(g_pos.x)
        adj_pos = _build_adj(g_pos.edge_index.cpu(), n_pos)
        tru     = set(g_pos.motif_id.nonzero(as_tuple=False).squeeze(-1).tolist())

        x_pos    = g_pos.x.numpy()
        real_pos = _sample(adj_pos, n_pos, args.n_samples, args.k_min, args.k_max)
        subs = []
        for ns in real_pos:
            if args.embed_mode == 'gae':
                x_t, ei_t = build_subgraph_tensors(ns, adj_pos, x_pos)
                with torch.no_grad():
                    z = gae_model.embed(x_t, ei_t).numpy()
            elif args.embed_mode == 'canonical':
                z = _iso_certificate(_to_nx(ns, adj_pos))
            else:
                z = embed_fn(_to_nx(ns, adj_pos), ns, dim=args.embed_dim, node_feats=x_pos)
            if args.embed_mode != 'canonical':
                Z_pos_list.append(z)
                Z_pos_sizes.append(len(ns))
            subs.append((ns, z))
        raw_graph_subs.append((tru, adj_pos, subs))

        if not args.load_model:
            g_neg    = pair['neg']
            n_neg    = len(g_neg.x)
            x_neg    = g_neg.x.numpy()
            adj_neg  = _build_adj(g_neg.edge_index.cpu(), n_neg)
            real_neg = _sample(adj_neg, n_neg, args.n_samples, args.k_min, args.k_max)
            neg_types = collections.Counter()
            for ns in real_neg:
                if args.embed_mode == 'gae':
                    x_t, ei_t = build_subgraph_tensors(ns, adj_neg, x_neg)
                    with torch.no_grad():
                        z = gae_model.embed(x_t, ei_t).numpy()
                elif args.embed_mode == 'canonical':
                    z = _iso_certificate(_to_nx(ns, adj_neg))
                    neg_types[z] += 1
                else:
                    z = embed_fn(_to_nx(ns, adj_neg), ns, dim=args.embed_dim, node_feats=x_neg)
                if args.embed_mode != 'canonical':
                    Z_neg_list.append(z)
                    Z_neg_sizes.append(len(ns))
            if args.embed_mode == 'canonical':
                neg_canon_per_graph.append(neg_types)

        if (idx + 1) % 100 == 0:
            print(f"  processed {idx+1}/{n_g}", flush=True)

    if args.embed_mode != 'canonical':
        Z_pos = np.array(Z_pos_list, dtype=np.float32)
        args.embed_dim = Z_pos.shape[1]

    tree_clf = None
    planes   = None

    if args.load_model:
        with open(args.load_model, 'rb') as f:
            ckpt = pickle.load(f)
        planes       = ckpt['planes']
        tree_clf     = ckpt.get('tree_clf')
        bucket_score = ckpt['bucket_score']
        r_pos        = ckpt['r_pos']
        r_neg        = ckpt['r_neg']
        print(f"  loaded model from {args.load_model}")
    else:
        Z_neg = np.array(Z_neg_list, dtype=np.float32)

        use_tree_clf = args.clf == 'tree' or (args.clf == 'auto' and args.embed_mode == 'tree')
        if args.embed_mode == 'canonical':
            # cross-graph average frequency scoring: score(type) = f_pos / (f_pos + f_neg)
            # where f_pos = mean count of type per pos graph (avg frequency, not binary presence)
            pos_canon_per_graph = [
                collections.Counter(z for _, z in subs) for _, _, subs in raw_graph_subs
            ]
            n_pos_g = len(pos_canon_per_graph)
            n_neg_g = len(neg_canon_per_graph)
            all_types = set(t for g in pos_canon_per_graph for t in g)
            r_pos = collections.Counter({
                t: sum(g[t] for g in pos_canon_per_graph) for t in all_types
            })
            r_neg = collections.Counter({
                t: sum(g[t] for g in neg_canon_per_graph) for t in all_types
            })
            alpha = args.score_alpha
            bucket_score = {
                t: (r_pos[t] / n_pos_g) ** alpha / ((r_pos[t] / n_pos_g) + (r_neg[t] / n_neg_g) + 1e-8)
                for t in r_pos
            }
            n_types = len(all_types)
            print(f"  isomorphism classes (przulj certificate): {n_types} distinct")
        elif use_tree_clf:
            from sklearn.tree import DecisionTreeClassifier
            X = np.vstack([Z_pos, Z_neg])
            y = np.array([1] * len(Z_pos_list) + [0] * len(Z_neg_list))
            tree_clf = DecisionTreeClassifier(
                max_depth=args.tree_depth, min_samples_leaf=20, random_state=42
            )
            tree_clf.fit(X, y)
            n_leaves = tree_clf.get_n_leaves()
            print(f"  tree depth={args.tree_depth}  leaves={n_leaves}")
            pos_leaves = tree_clf.apply(Z_pos).tolist()
            neg_leaves = tree_clf.apply(Z_neg).tolist()
            r_pos = collections.Counter(pos_leaves)
            r_neg = collections.Counter(neg_leaves)
        else:
            if args.proj == 'lda':
                planes, separation = _fit_planes(Z_pos, Z_neg, args.hash_dim, args.embed_dim)
                print(f"  lda class separation (fisher): {separation:.3f}")
            else:
                n_components = min(args.hash_dim, Z_pos.shape[0], Z_pos.shape[1])
                pca = PCA(n_components=n_components, random_state=42)
                pca.fit(Z_pos)
                planes = pca.components_
                print(f"  pca variance explained: {pca.explained_variance_ratio_.sum():.3f}")

            r_pos = collections.defaultdict(int)
            r_neg = collections.defaultdict(int)
            for z in Z_pos_list:
                r_pos[_simhash(z, planes)] += 1
            for z in Z_neg_list:
                r_neg[_simhash(z, planes)] += 1

        if args.embed_mode == 'canonical':
            pass  # bucket_score already set above from cross-graph prevalence
        elif args.score_mode == 'knn':
            # per-subgraph knn density ratio, computed within each k-size group.
            # comparing only same-size subgraphs avoids size confounding wl embeddings.
            # d_neg/d_pos: large when dense in pos cloud, sparse in neg cloud.
            # scores are then averaged per lsh bucket (analogous to averaging sigma
            # per bucket in motifiesta's hashdecoder).
            # reference trees are built on a random subsample (ref_size) for speed;
            # queries still run over the full set so every subgraph gets a score.
            _REF_SIZE = 10_000
            pos_arr   = np.array(Z_pos_list, dtype=np.float32)
            neg_arr   = np.array(Z_neg_list, dtype=np.float32)
            pos_sizes = np.array(Z_pos_sizes)
            neg_sizes = np.array(Z_neg_sizes)
            sub_scores = np.zeros(len(Z_pos_list), dtype=np.float32)
            for ksz in np.unique(pos_sizes):
                pm = pos_sizes == ksz
                nm = neg_sizes == ksz
                if pm.sum() < 2 or nm.sum() < 2:
                    sub_scores[pm] = 1.0
                    continue
                p_idx = np.where(pm)[0]
                n_idx = np.where(nm)[0]
                p_ref = pos_arr[np.random.choice(p_idx, min(_REF_SIZE, len(p_idx)), replace=False)]
                n_ref = neg_arr[np.random.choice(n_idx, min(_REF_SIZE, len(n_idx)), replace=False)]
                k = min(args.knn_k, len(p_ref) - 1, len(n_ref) - 1)
                kd_pos = KDTree(p_ref)
                kd_neg = KDTree(n_ref)
                d_pos  = kd_pos.query(pos_arr[pm], k=k + 1)[0][:, -1]
                d_neg  = kd_neg.query(pos_arr[pm], k=k)[0][:, -1]
                sub_scores[pm] = d_neg / (d_pos + 1e-8)
            bucket_sums   = collections.defaultdict(float)
            bucket_counts = collections.defaultdict(int)
            for z, s in zip(Z_pos_list, sub_scores):
                h = _simhash(z, planes)
                bucket_sums[h]   += float(s)
                bucket_counts[h] += 1
            bucket_score = {h: bucket_sums[h] / bucket_counts[h] for h in bucket_sums}
            print(f"  knn scoring (k={args.knn_k}): {len(bucket_score)} buckets")
        elif args.score_mode == 'gstat':
            import math
            n_pos_total = sum(r_pos.values())
            n_neg_total = sum(r_neg.values()) or 1
            N = n_pos_total + n_neg_total
            base_rate = n_pos_total / N
            def _gstat(rp, rn):
                n = rp + rn
                ep = n * base_rate
                en = n * (1 - base_rate)
                g = 0.0
                if rp > 0 and ep > 0:
                    g += rp * math.log(rp / ep)
                if rn > 0 and en > 0:
                    g += rn * math.log(rn / en)
                return 2.0 * g
            bucket_score = {
                h: _gstat(r_pos[h], r_neg.get(h, 0))
                for h in r_pos
            }
        elif args.embed_mode != 'canonical':
            alpha = args.score_alpha
            bucket_score = {
                h: r_pos[h] ** alpha / (r_pos[h] + r_neg.get(h, 0) + 1e-8)
                for h in r_pos
            }

        if args.save_model:
            ckpt = {
                'planes':       planes,
                'tree_clf':     tree_clf,
                'bucket_score': bucket_score,
                'r_pos':        dict(r_pos),
                'r_neg':        dict(r_neg),
            }
            with open(args.save_model, 'wb') as f:
                pickle.dump(ckpt, f)
            print(f"  saved model to {args.save_model}")

    def _bucket(z):
        if args.embed_mode == 'canonical':
            return z  # canonical type tuple is the bucket key directly
        if tree_clf is not None:
            return int(tree_clf.apply(z.reshape(1, -1))[0])
        return _simhash(z, planes)

    # assign each pos subgraph to its bucket for inference
    graph_subs = []
    true_sets  = []
    adjs       = []
    for tru, adj_pos, subs in raw_graph_subs:
        true_sets.append(tru)
        adjs.append(adj_pos)
        g_buckets = []
        for ns, z in subs:
            g_buckets.append((ns, _bucket(z)))
        graph_subs.append(g_buckets)

    top_buckets = sorted(bucket_score, key=lambda h: (bucket_score[h], r_pos[h]), reverse=True)[:args.top_k]

    # print bucket landscape for top candidates
    print(f"\n  top-{args.top_k} bucket candidates (freq x purity):")
    for h in top_buckets:
        rp, rn = r_pos[h], r_neg.get(h, 0)
        purity = rp / (rp + rn + 1e-8)
        print(f"    pos={rp:4d}  neg={rn:4d}  purity={purity:.3f}"
              f"  score={bucket_score[h]:.1f}")

    best_jaccard = -1.0

    if args.pred_mode == 'subgraph':
        # score each subgraph individually by its bucket score, prefer larger k on ties.
        # take top-k subgraphs per graph and union their nodes.
        total_inter = 0
        total_union = 0
        for i, g_buckets in enumerate(graph_subs):
            scored = sorted(
                [(bucket_score.get(h, 0.0), len(ns), ns) for ns, h in g_buckets],
                key=lambda x: (x[0], x[1]),
                reverse=True,
            )
            pred = {n for _, _, ns in scored[:args.top_k] for n in ns}
            if args.cc_filter and pred:
                pred = _largest_component(pred, adjs[i])
            tru = true_sets[i]
            total_inter += len(pred & tru)
            total_union += len(pred | tru)
        best_jaccard = total_inter / max(total_union, 1)
        print(f"\njaccard:     {best_jaccard:.4f}  (pred_mode=subgraph  top_k={args.top_k})")
    else:
        best_bucket = top_buckets[0]
        for cand in top_buckets:
            total_inter = 0
            total_union = 0
            for i, g_buckets in enumerate(graph_subs):
                pred = {n for ns, h in g_buckets if h == cand for n in ns}
                if args.cc_filter and pred:
                    pred = _largest_component(pred, adjs[i])
                tru  = true_sets[i]
                total_inter += len(pred & tru)
                total_union += len(pred | tru)
            j = total_inter / max(total_union, 1)
            if j > best_jaccard:
                best_jaccard = j
                best_bucket  = cand
        print(f"\njaccard:     {best_jaccard:.4f}")
        print(f"best_bucket: pos={r_pos[best_bucket]}"
              f"  neg={r_neg.get(best_bucket, 0)}"
              f"  purity={r_pos[best_bucket]/(r_pos[best_bucket]+r_neg.get(best_bucket,0)+1e-8):.3f}"
              f"  score={bucket_score[best_bucket]:.1f}")


if __name__ == '__main__':
    main()
    sys.exit(0)
