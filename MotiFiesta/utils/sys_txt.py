import torch
import pandas as pd
import os.path as osp
from torch_geometric.data import Data, Dataset

class SysTxtDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['mips.txt']

    @property
    def processed_file_names(self):
        # cache both the real and the rewired null graph
        return ['real_graph.pt', 'null_graph.pt']

    def process(self):
        # load the raw mips data
        # sep=none handles mixed whitespace automatically
        df = pd.read_csv(self.raw_paths[0], sep=None, engine='python', comment='#', header=None)
        
        # establish the master node list from both columns to cover all proteins
        # convert to strings for consistent mapping
        nodes = pd.concat([df[0], df[1]]).astype(str).unique()
        node_map = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)
        
        # build the edge index for the graph topology
        src = [node_map[s] for s in df[0]]
        dst = [node_map[d] for d in df[1]]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # feature engineering: map biological signal to 25 dimensions
        # initialize with small noise to ensure the decoder doesn't collapse
        x = torch.randn((num_nodes, 25)) * 0.1
        
        # create a unique lookup map for protein sizes and categories
        # take the first instance of each protein to avoid duplicate label errors
        feature_map = df[[0, 2, 4]].drop_duplicates(subset=[0]).set_index(0)

        # map features to nodes and fill missing values (orphans) with 0
        node_to_size = feature_map[4].reindex(nodes).fillna(0)
        node_to_cat = feature_map[2].astype(str).apply(hash).reindex(nodes).fillna(0)

        # fill the first two slots of 25-dim tensor
        x[:, 0] = torch.tensor(node_to_size.values, dtype=torch.float) / 100.0
        x[:, 1] = torch.tensor(node_to_cat.values, dtype=torch.float) % 10

        # create the final data objects covering all 8617 nodes
        real_data = Data(x=x, edge_index=edge_index)
        real_data.node_names = list(nodes)
        
        # create a null graph for the contrastive loss function
        # shuffle the edge index to create a randomized version of the graph
        null_edge_index = edge_index[:, torch.randperm(edge_index.size(1))]
        null_data = Data(x=x, edge_index=null_edge_index)

        # save the processed data to the disk cache
        torch.save(real_data, self.processed_paths[0])
        torch.save(null_data, self.processed_paths[1])

    def len(self):
        return 1

    def get(self, idx):
        # returns pos & neg as a tuple for the loader to ingest
        real = torch.load(self.processed_paths[0])
        null = torch.load(self.processed_paths[1])
        #return real, null
        return {
            'pos': real,
            'neg': null
        }