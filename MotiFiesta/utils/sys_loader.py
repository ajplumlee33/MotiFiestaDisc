from torch_geometric.loader import NeighborLoader
import torch


class SysLoader:
    """
    lightweight wrapper around NeighborLoader that yields {'pos': batch} dicts.
    null (rewired) samples are generated downstream in sys_train when the motif
    phase kicks in, so the loader pays no cost during warmup.
    """
    def __init__(self, data, batch_size=128, input_nodes=None, num_neighbors=[10, 5], **kwargs):
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
            yield {'pos': batch}

    def __len__(self):
        return len(self.loader)
