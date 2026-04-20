import torch
import pandas as pd
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_networkx
from torch_geometric.utils import degree


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
        # store source graph as an attribute so the model can query it for "truth"
        self.nx_graph = to_networkx(self.cached_data, to_undirected=True)

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
        # load raw edge list
        # handles tabs or spaces automatically
        df = pd.read_csv(self.raw_paths[0], sep=None, engine='python', comment='#', header=None)

        # establish universal node list
        # must include both columns to ensure 0-indexed continuity
        nodes = pd.concat([df[0], df[1]]).astype(str).unique()
        node_map = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)

        # map edges to these indices
        src = df[0].astype(str).map(node_map).values
        dst = df[1].astype(str).map(node_map).values
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # enforce undirected connectivity
        edge_index = to_undirected(edge_index)

        # determine the one-hot degree cap if not provided
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
        max_deg = self.max_degree if self.max_degree is not None else int(deg.max().item())

        # build data object and apply one-hot degree encoding
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        data = T.OneHotDegree(max_deg)(data)

        torch.save(data, self.processed_paths[0])

    def len(self):
        # returns 1 for the single system graph
        return 1

    def get(self, idx):
        return self.cached_data
