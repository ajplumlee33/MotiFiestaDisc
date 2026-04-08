from itertools import combinations
from itertools import starmap
import numpy as np
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl
import torch
from igraph import Graph
from wwl import wwl
from wwl.propagation_scheme import WeisfeilerLehman
from wwl.wwl import pairwise_wasserstein_distance
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram

def build_K(subgraphs, cache=None):
    graph_pairs = ((*c, cache) for c in combinations(subgraphs, 2))
    d = list(starmap(subgraph_sim_dgl, graph_pairs))
    N = len(subgraphs)

    block = np.zeros((N, N))
    block[np.triu_indices(N, 1)] = d
    block += block.T
    block += np.eye(N)

    return torch.tensor(block, dtype=torch.float)

""" def build_wwl_K(graphs, node_features=None):
    graphs = [Graph.from_networkx(g) for g in graphs]
    kernel_matrix = wwl(graphs,
                        node_features,
                        num_iterations=4
                        )
    return torch.tensor(kernel_matrix, dtype=torch.float) """

""" def build_wwl_K(graphs, node_features):
    
    Computes the WWL kernel matrix by iterating through pairs.
    This bypasses the ragged array error and internal API naming issues.
    
    if not graphs:
        return torch.zeros((0, 0))

    n = len(graphs)
    # Initialize the similarity matrix
    kernel_matrix = np.zeros((n, n), dtype=np.float64)

    # Compute the pairwise similarities
    for i in range(n):
        for j in range(i, n):
            # Pass only TWO motifs at a time
            # Even with different sizes, a pair is much safer for the library
            pair_graphs = [graphs[i], graphs[j]]
            pair_feats = [node_features[i], node_features[j]]
            
            try:
                # wwl returns a 2x2 similarity matrix for the pair
                k_pair = wwl(pair_graphs, node_features=pair_feats)
                val = k_pair[0, 1]
            except Exception:
                # Fallback if a specific pair is too small/isolated
                val = 0.0
            
            kernel_matrix[i, j] = val
            kernel_matrix[j, i] = val # Symmetry

    return torch.from_numpy(kernel_matrix).float() """

""" def build_wwl_K(graphs, node_features=None):
    graphs = [Graph.from_networkx(g) for g in graphs]
    kernel_matrix = wwl(graphs,
                        node_features,
                        num_iterations=4
                        )
    return torch.tensor(kernel_matrix, dtype=torch.float) """

def build_wwl_K(graphs, node_features):
    """
    standard grakel approach for variable-sized attributed motifs.
    """
    grakel_graphs = []
    
    for i, g in enumerate(graphs):
        edges = list(g.edges())
        
        # ensure features are a numpy array
        feats = np.array(node_features[i], dtype=np.float64)
         
        # for WL requires labels to be hashable
        attrs = {n: tuple(feats[j]) for j, n in enumerate(g.nodes())}
        
        grakel_graphs.append(Graph(edges, node_labels=attrs))

    # weisfeilerlehman uses vertexhistogram as the base kernel
    # vertexhistogram will now handle the hashable tuples
    wl_kernel = WeisfeilerLehman(base_graph_kernel=VertexHistogram, n_iter=4, normalize=True)

    # compute the full k matrix
    kernel_matrix = wl_kernel.fit_transform(grakel_graphs)

    return torch.from_numpy(kernel_matrix).float()

def subgraph_sim(sg1, sg2, timeout=1):
    return nx.algorithms.graph_edit_distance(sg1, sg2, timeout=timeout)

def subgraph_sim_dgl(sg1, sg2, cache, beta=.5, node_attr=None, edge_attr=None):
    G1 = dgl.from_networkx(sg1)
    G2 = dgl.from_networkx(sg2)

    node_sub, edge_sub = (None, None)

    if not node_attr is None:
        node_sub = build_sub_matrix(G1, G2, node_attr, mode='node')
    if not edge_attr is None:
        edge_sub = build_sub_matrix(G1, G2, edge_attr, mode='edge')

    if not cache is None:
        h_g1 = wl(sg1)
        h_g2 = wl(sg2)
        h_g1, h_g2 = sorted([h_g1, h_g2])
        try:
            distance = cache[h_g1][h_g2]
        except KeyError:
            try:
                distance,_,_ = graph_edit_distance(G1,
                                                   G2,
                                                   algorithm='hausdorff',
                                                   node_substitution_cost=node_sub,
                                                   edge_substitution_cost=edge_sub
                                                   )
            except Exception as e:
                print(e)
                print(G1.nodes(), G2.nodes())
                distance = float(abs(len(G1.nodes()) - len(G2.nodes())))
            cache[h_g1][h_g2] = distance
    else:
            distance,_,_ = graph_edit_distance(G1, G2, algorithm='hausdorff')

    sim = np.exp(-beta * distance)
    return sim
