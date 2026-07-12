"""shared rand-esu evaluation utilities: sampling, projection, fitting, scoring, jaccard."""
import collections, math, pickle, random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from torch_geometric.utils import remove_self_loops, coalesce


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

def sample(adj, n_nodes, n_samples, k_min, k_max):
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

def build_adj(ei, n):
    ei, _ = remove_self_loops(ei)
    ei, _ = coalesce(ei, None, num_nodes=n)
    a = [[] for _ in range(n)]
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        a[u].append(v)
    return a

def largest_component(nodes, adj):
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


# ─── projection ───────────────────────────────────────────────────────────────

def simhash(z, planes):
    return tuple((planes @ z > 0).astype(np.int8).tolist())

def fit_planes(Z_pos, Z_neg, hash_dim, embed_dim):
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


# ─── scoring ──────────────────────────────────────────────────────────────────

def score_gstat(r_pos, r_neg):
    n_pos_total = sum(r_pos.values())
    n_neg_total = sum(r_neg.values()) or 1
    N = n_pos_total + n_neg_total
    base_rate = n_pos_total / N
    def _g(rp, rn):
        n = rp + rn
        ep, en = n * base_rate, n * (1 - base_rate)
        g = 0.0
        if rp > 0 and ep > 0: g += rp * math.log(rp / ep)
        if rn > 0 and en > 0: g += rn * math.log(rn / en)
        return 2.0 * g
    return {h: _g(r_pos[h], r_neg.get(h, 0)) for h in r_pos}

def score_count(r_pos, r_neg, alpha):
    return {h: r_pos[h] ** alpha / (r_pos[h] + r_neg.get(h, 0) + 1e-8) for h in r_pos}

def score_knn(Z_pos_list, Z_neg_list, Z_pos_sizes, Z_neg_sizes, planes, knn_k):
    pos_arr, neg_arr   = np.array(Z_pos_list, dtype=np.float32), np.array(Z_neg_list, dtype=np.float32)
    pos_sizes, neg_sizes = np.array(Z_pos_sizes), np.array(Z_neg_sizes)
    sub_scores = np.zeros(len(Z_pos_list), dtype=np.float32)
    _REF = 10_000
    for ksz in np.unique(pos_sizes):
        pm, nm = pos_sizes == ksz, neg_sizes == ksz
        if pm.sum() < 2 or nm.sum() < 2:
            sub_scores[pm] = 1.0; continue
        p_ref = pos_arr[np.random.choice(np.where(pm)[0], min(_REF, pm.sum()), replace=False)]
        n_ref = neg_arr[np.random.choice(np.where(nm)[0], min(_REF, nm.sum()), replace=False)]
        k = min(knn_k, len(p_ref) - 1, len(n_ref) - 1)
        d_pos = KDTree(p_ref).query(pos_arr[pm], k=k + 1)[0][:, -1]
        d_neg = KDTree(n_ref).query(pos_arr[pm], k=k)[0][:, -1]
        sub_scores[pm] = d_neg / (d_pos + 1e-8)
    sums, counts = collections.defaultdict(float), collections.defaultdict(int)
    for z, s in zip(Z_pos_list, sub_scores):
        h = simhash(z, planes); sums[h] += float(s); counts[h] += 1
    return {h: sums[h] / counts[h] for h in sums}


# ─── fitting ──────────────────────────────────────────────────────────────────

def fit(Z_pos_list, Z_neg_list, Z_pos_sizes, Z_neg_sizes,
        pos_canon_per_graph, neg_canon_per_graph, args):
    """fit classifier and compute bucket scores. returns (planes, tree_clf, r_pos, r_neg, bucket_score)."""
    from sklearn.tree import DecisionTreeClassifier

    Z_pos = np.array(Z_pos_list, dtype=np.float32) if Z_pos_list else None
    Z_neg = np.array(Z_neg_list, dtype=np.float32) if Z_neg_list else None
    tree_clf = None
    planes   = None

    if args.embed_mode == 'canonical':
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
        print(f"  isomorphism classes (przulj certificate): {len(all_types)} distinct")
        return planes, tree_clf, r_pos, r_neg, bucket_score

    if args.clf == 'tree':
        X = np.vstack([Z_pos, Z_neg])
        y = np.array([1] * len(Z_pos_list) + [0] * len(Z_neg_list))
        tree_clf = DecisionTreeClassifier(
            max_depth=args.tree_depth, min_samples_leaf=20, random_state=42
        )
        tree_clf.fit(X, y)
        print(f"  tree depth={args.tree_depth}  leaves={tree_clf.get_n_leaves()}")
        r_pos = collections.Counter(tree_clf.apply(Z_pos).tolist())
        r_neg = collections.Counter(tree_clf.apply(Z_neg).tolist())
        bucket_score = score_gstat(r_pos, r_neg) if args.score == 'gstat' else score_count(r_pos, r_neg, args.score_alpha)
    else:
        planes, separation = fit_planes(Z_pos, Z_neg, args.hash_dim, args.embed_dim)
        print(f"  lda class separation (fisher): {separation:.3f}")
        r_pos = collections.defaultdict(int)
        r_neg = collections.defaultdict(int)
        for z in Z_pos_list: r_pos[simhash(z, planes)] += 1
        for z in Z_neg_list: r_neg[simhash(z, planes)] += 1
        if args.clf == 'knn':
            bucket_score = score_knn(Z_pos_list, Z_neg_list, Z_pos_sizes, Z_neg_sizes, planes, args.knn_k)
            print(f"  knn scoring (k={args.knn_k}): {len(bucket_score)} buckets")
        else:
            bucket_score = score_gstat(r_pos, r_neg) if args.score == 'gstat' else score_count(r_pos, r_neg, args.score_alpha)

    return planes, tree_clf, r_pos, r_neg, bucket_score


def save_model(path, planes, tree_clf, bucket_score, r_pos, r_neg):
    with open(path, 'wb') as f:
        pickle.dump({'planes': planes, 'tree_clf': tree_clf,
                     'bucket_score': bucket_score, 'r_pos': dict(r_pos), 'r_neg': dict(r_neg)}, f)
    print(f"  saved model to {path}")

def load_model(path):
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    print(f"  loaded model from {path}")
    return ckpt['planes'], ckpt.get('tree_clf'), ckpt['r_pos'], ckpt['r_neg'], ckpt['bucket_score']


# ─── jaccard evaluation ───────────────────────────────────────────────────────

def eval_jaccard(graph_subs, true_sets, adjs, bucket_score, r_pos, top_k, pred_mode, cc_filter):
    top_buckets = sorted(bucket_score, key=lambda h: (bucket_score[h], r_pos[h]), reverse=True)[:top_k]

    print(f"\n  top-{top_k} bucket candidates (freq x purity):")
    for h in top_buckets:
        rp = r_pos[h]
        print(f"    pos={rp:4d}  score={bucket_score[h]:.1f}")

    best_jaccard = -1.0
    if pred_mode == 'subgraph':
        total_inter = total_union = 0
        for i, g_buckets in enumerate(graph_subs):
            scored = sorted(
                [(bucket_score.get(h, 0.0), len(ns), ns) for ns, h in g_buckets],
                key=lambda x: (x[0], x[1]), reverse=True,
            )
            pred = {n for _, _, ns in scored[:top_k] for n in ns}
            if cc_filter and pred:
                pred = largest_component(pred, adjs[i])
            total_inter += len(pred & true_sets[i])
            total_union += len(pred | true_sets[i])
        best_jaccard = total_inter / max(total_union, 1)
        print(f"\njaccard:     {best_jaccard:.4f}  (pred_mode=subgraph  top_k={top_k})")
    else:
        best_bucket = top_buckets[0]
        for cand in top_buckets:
            total_inter = total_union = 0
            for i, g_buckets in enumerate(graph_subs):
                pred = {n for ns, h in g_buckets if h == cand for n in ns}
                if cc_filter and pred:
                    pred = largest_component(pred, adjs[i])
                total_inter += len(pred & true_sets[i])
                total_union += len(pred | true_sets[i])
            j = total_inter / max(total_union, 1)
            if j > best_jaccard:
                best_jaccard = j
                best_bucket  = cand
        print(f"\njaccard:     {best_jaccard:.4f}")
        print(f"best_bucket: pos={r_pos[best_bucket]}  score={bucket_score[best_bucket]:.1f}")

    return best_jaccard
