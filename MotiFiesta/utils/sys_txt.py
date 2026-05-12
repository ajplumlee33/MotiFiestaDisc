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
from MotiFiesta.utils.graph_utils import torch_double_edge_swap


def rewire(g_pyg, n_iter=100):
    """fully-vectorized double edge swap. drop-in replacement for the
    networkx-based version. preserves node features (x) and degree
    sequence exactly. drops the connectivity-preservation check —
    fine for negative samples used in contrastive losses.

    keeps the n_iter parameter name for call-site compatibility; each
    iteration in the original corresponded to one swap, so n_iter maps
    directly to n_swaps in the vectorized version.
    """
    new_edge_index = torch_double_edge_swap(g_pyg.edge_index, n_swaps=n_iter)
    rewired_pyg = Data(x=g_pyg.x, edge_index=new_edge_index)
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
