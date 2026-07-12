"""subgraph embedding functions for rand-esu disc experiments."""
import networkx as nx
import numpy as np


_LABEL_VEC_CACHE: dict = {}


def _label_vec(label, dim):
    key = (label, dim)
    if key not in _LABEL_VEC_CACHE:
        rng = np.random.RandomState(abs(hash(label)) % (2**31))
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        _LABEL_VEC_CACHE[key] = v
    return _LABEL_VEC_CACHE[key]


def to_nx(nodes, adj):
    nm = {v: i for i, v in enumerate(nodes)}
    G  = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for u in nodes:
        for v in adj[u]:
            if v in nm and nm[u] < nm[v]:
                G.add_edge(nm[u], nm[v])
    return G


def wl_embed(nx_graph, nodes, num_iter=4, dim=32, node_feats=None, **kwargs):
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


def wwl_embed(nx_graph, nodes, num_iter=4, dim=32, node_feats=None, **kwargs):
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



def tree_embed(nx_graph, nodes=None, dim=None, node_feats=None, **kwargs):
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


def canonical_cert(nx_graph, nodes=None, **kwargs):
    """przulj isomorphism class certificate via sorted per-node (degree, triangle) pairs."""
    tri = nx.triangles(nx_graph)
    return tuple(sorted((d, tri[n]) for n, d in nx_graph.degree()))


EMBED_FNS = {
    'wl':        wl_embed,
    'wwl':       wwl_embed,
    'tree':      tree_embed,
    'canonical': canonical_cert,
}
