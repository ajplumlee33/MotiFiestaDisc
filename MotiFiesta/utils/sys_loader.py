import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader


MAX_DEGREE = 25


class SysLoader:
    """
    lightweight wrapper around NeighborLoader that yields {'pos': batch} dicts.
    adds degree one-hot node features when the source graph has no x.
    """
    def __init__(self, data, batch_size=128, input_nodes=None, num_neighbors=None, **kwargs):
        if num_neighbors is None:
            num_neighbors = [10, 5]
        # pre-compute degree one-hot on the full graph if no features exist
        if data.x is None:
            deg = data.edge_index[0].bincount(minlength=data.num_nodes).clamp(max=MAX_DEGREE - 1)
            data.x = F.one_hot(deg, num_classes=MAX_DEGREE).float()
        self.loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=input_nodes,
            subgraph_type='induced',
            **kwargs
        )

    def __iter__(self):
        for batch in self.loader:
            if not hasattr(batch, 'batch') or batch.batch is None:
                batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)
            # recompute degree features from local subgraph to avoid global-degree leakage
            deg = batch.edge_index[0].bincount(minlength=batch.num_nodes).clamp(max=MAX_DEGREE - 1)
            batch.x = F.one_hot(deg, num_classes=MAX_DEGREE).float()
            yield {'pos': batch}

    def __len__(self):
        return len(self.loader)
