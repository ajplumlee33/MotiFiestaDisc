import random

import torch
import pandas as pd
import networkx as nx

from igraph import Graph
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx
from torch_geometric.utils import degree


def rewire(g_pyg, n_iter=100):
    """ Apply (u, v), (u', v') --> (u, v'), (v, u') to randomize the graph.

    mirrors real_world.rewire: operates on a pyg Data, routes through networkx
    for the classical double edge swap, preserves node features via the `x`
    attribute. returns a new pyg Data with the rewired edge set.
    """
    has_features = g_pyg.x is not None
    if has_features:
        g_nx = to_networkx(g_pyg, node_attrs=['x'])
    else:
        g_nx = to_networkx(g_pyg)
    rewired_g = g_nx.copy()
    # cache once — list(g_nx.edges()) inside the loop is O(E) per iteration
    edge_list = list(g_nx.edges())
    for n in range(n_iter):
        e1, e2 = random.sample(edge_list, 2)
        rewired_g.remove_edges_from([e1, e2])
        rewired_g.add_edges_from([(e1[0], e2[1]), (e1[1], e2[0])])

    rewired_g.remove_edges_from(list(nx.selfloop_edges(rewired_g)))
    if has_features:
        rewired_pyg = from_networkx(rewired_g, group_node_attrs=['x'])
    else:
        rewired_pyg = from_networkx(rewired_g)
    return rewired_pyg


class SysTxtDataset(Dataset):
    def __init__(self, root, max_degree=None, n_features=None, transform=None, pre_transform=None):
        """ Builds a single-graph dataset from a plain edge list file.

        Args:
        ---
        root (str): path to folder containing the raw edge list
        max_degree (int): cap for one-hot degree encoding. if None, uses the
            graph's actual max degree at process time.
        n_features (int): optional override for num_features
        """
        self.max_degree = max_degree
        self.n_features = n_features
        super().__init__(root, transform, pre_transform)

        # load into memory once
        self.cached_data = torch.load(self.processed_paths[0], weights_only=False)
        # store source graph as an igraph Graph for fast subgraph extraction
        edges = self.cached_data.edge_index.t().tolist()
        self.ig_graph = Graph(n=self.cached_data.num_nodes, edges=edges, directed=False)
        self.ig_graph.simplify()

    @property
    def raw_file_names(self):
        return ['mips.txt']

    @property
    def processed_file_names(self):
        return ['system_graph.pt']

    @property
    def num_features(self):
        if self.n_features is not None:
            return self.n_features
        return self.cached_data.num_features

    def process(self):
        df = pd.read_csv(self.raw_paths[0], sep=None, engine='python', comment='#', header=None)

        nodes = pd.concat([df[0], df[1]]).astype(str).unique()
        node_map = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)

        src = df[0].astype(str).map(node_map).values
        dst = df[1].astype(str).map(node_map).values
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        edge_index = to_undirected(edge_index)

        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
        max_deg = self.max_degree if self.max_degree is not None else int(deg.max().item())

        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        data = T.OneHotDegree(max_deg)(data)

        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        return self.cached_data
