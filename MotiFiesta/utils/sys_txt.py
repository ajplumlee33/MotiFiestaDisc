import torch
import pandas as pd
import numpy as np
import os.path as osp
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_networkx

class SysTxtDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # load into memory once
        self.cached_data = torch.load(self.processed_paths[0], weights_only=False)
        # store master graph as an attribute so the model can query it for "truth"
        self.nx_graph = to_networkx(self.cached_data, to_undirected=True)

    @property
    def raw_file_names(self):
        return ['mips.txt']

    @property
    def processed_file_names(self):
        return ['system_graph.pt']

    def process(self):
        # load raw mips ppi data
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
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
        
        # enforce undirected connectivity for ppi consistency
        edge_index = to_undirected(edge_index)

        # feature engineering (rigid 25-dim structure)
        # using float32 to prevent numpy 'object' type conversion errors
        x = torch.zeros((num_nodes, 25), dtype=torch.float32)
        
        # build a complete metadata map (column 2 = cat, column 4 = size)
        m1 = df[[0, 2, 4]].rename(columns={0: 'prot', 2: 'cat', 4: 'size'})
        m2 = df[[1, 2, 4]].rename(columns={1: 'prot', 2: 'cat', 4: 'size'})
        meta_master = pd.concat([m1, m2]).drop_duplicates(subset=['prot']).set_index('prot')

        # map using reindex to guarantee alignment with node_map
        sizes = meta_master['size'].reindex(nodes).fillna(0).values
        cats = meta_master['cat'].astype(str).apply(hash).reindex(nodes).fillna(0).values

        x[:, 0] = torch.from_numpy(sizes).float() / 100.0
        x[:, 1] = torch.from_numpy(cats % 10).float()
        
        # add some noise for remaining 23 dimensions
        # this ensures the matrix is dense and prevents singular kernels
        x[:, 2:] = torch.randn((num_nodes, 23)) * 0.01

        # create and save
        data = Data(x=x, edge_index=edge_index)
        torch.save(data, self.processed_paths[0])

    def len(self):
        # returns 1 for the single system graph
        return 1

    def get(self, idx):
        return self.cached_data