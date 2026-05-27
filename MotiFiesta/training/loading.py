import os
import math

import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from MotiFiesta.utils.synthetic import SyntheticMotifs
from MotiFiesta.utils.real_world import RealWorldDataset
from MotiFiesta.utils.sys_txt import SysTxtDataset
from MotiFiesta.utils.sys_synthetic import SysSyntheticDataset
from MotiFiesta.utils.sys_loader import SysLoader
from MotiFiesta.utils.bfs_decompose import BFSDecomposedDataset, LouvainDecomposedDataset


def get_loader(root,
               name='synthetic',
               batch_size=2,
               **kwargs
               ):
    """
    Arguments
    ----------
    root:
        path to folder for storing the dataset
    name:
        ID of dataset (options: 'synthetic' generates synthetic motifs, else the string ID of a PyG dataset

    Returns
    -------

    dict:
        Dictionary with loaders and datasets for train/test

    """
    if 'bfs_decomp' in root or 'louvain_decomp' in root:
        cache = os.path.join(root, 'processed', 'data_0.pt')
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"no decomposed cache at {cache} — run scripts/build_data_bfs.py first"
            )
        if 'louvain_decomp' in root:
            print(">>> SUCCESS: Louvain-decomposed dataset detected")
            dataset = LouvainDecomposedDataset(root=root)
        else:
            print(">>> SUCCESS: BFS-decomposed dataset detected")
            dataset = BFSDecomposedDataset(root=root)
    elif 'mips_torch' in root:
        print(">>> SUCCESS: Systems dataset detected")
        dataset = SysTxtDataset(root=root)
    elif 'sys_synth' in root:
        # NEW: single-graph synthetic dataset with planted motif ground truth.
        # Extract only the kwargs this class actually accepts — callers like
        # the CLI pass `attributed=...` and other flags meant for other
        # branches, and forwarding them blindly would TypeError here.
        print(">>> SUCCESS: Synthetic systems dataset detected")
        _synth_keys = ('motif_type', 'motif_size', 'n_motifs',
                       'parent_size', 'parent_e_prob', 'distort_p',
                       'seed')
        _synth_kwargs = {k: kwargs[k] for k in _synth_keys if k in kwargs}
        dataset = SysSyntheticDataset(root=root, **_synth_kwargs)
    else:
        print(">>> FAIL: Falling back to RealWorldDataset")
        if not name.startswith('synth'):
            if name == 'IMDB-BINARY':
                dataset = RealWorldDataset(root=root, max_degree=300, n_features=301)
            else:
                dataset = RealWorldDataset(root=root)
        else:
            dataset = SyntheticMotifs(root=root, name=name, **kwargs)
    if len(dataset) <= 1:
        # no split for a systems-level graph
        # same dataset for both, masks for train/test
        print("systems-level graph detected: skipping dataset split")
        # calculate split indices
        num_nodes = dataset[0].num_nodes
        indices = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(42))
        split_idx = int(num_nodes * 0.8)

        # degree-biased seed sampling: high-degree nodes (motif nodes) are
        # sampled as seeds proportionally more often, so training batches are
        # more frequently centered on motif instances. test and decode loaders
        # stay uniform so evaluation is unbiased.
        data = dataset[0]
        degrees = data.edge_index[0].bincount(minlength=num_nodes).float()
        train_idx = torch.multinomial(
            degrees + 1.0,
            num_samples=split_idx,
            replacement=True,
            generator=torch.Generator().manual_seed(42),
        )
        test_idx = indices[split_idx:]

        # num_neighbors matches motif_size so each batch centered on a motif
        # seed captures exactly one complete motif instance at hop-1.
        motif_size = int(kwargs.get('motif_size', 10))
        num_neighbors = [motif_size - 1, 0]

        loader_train = SysLoader(dataset[0], input_nodes=train_idx, batch_size=batch_size, num_neighbors=num_neighbors, shuffle=True)
        loader_test = SysLoader(dataset[0], input_nodes=test_idx, batch_size=batch_size, num_neighbors=num_neighbors, shuffle=True)
        loader = SysLoader(dataset[0], batch_size=batch_size, num_neighbors=num_neighbors, shuffle=False)
    else:
        lengths = [math.floor(len(dataset) * .8), math.ceil(len(dataset) * .2)]
        train_data, test_data = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
        loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return {'dataset_whole': dataset, 'loader_whole': loader, 'loader_train': loader_train, 'loader_test': loader_test}


if __name__ == "__main__":
    data = get_loader(root='barbell-pair')
    print(data)
