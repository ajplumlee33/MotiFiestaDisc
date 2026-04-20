from torch_geometric.loader import NeighborLoader
import torch


def fast_vectorized_swap(edge_index, swap_ratio=0.5):
    """
    native pytorch degree-preserving swap
    preserves source node degrees (row)
    """
    row = edge_index[0]
    col = edge_index[1]
    num_edges = row.size(0)

    mask = torch.rand(num_edges, device=edge_index.device) < swap_ratio
    swap_idx = torch.where(mask)[0]

    if len(swap_idx) > 1:
        new_col = col.clone()
        # shuffle the target nodes for the selected edges
        shuffled_indices = torch.randperm(len(swap_idx), device=edge_index.device)
        new_col[swap_idx] = col[swap_idx][shuffled_indices]
        return torch.stack([row, new_col], dim=0)

    return edge_index


class SysLoader:
    def __init__(self, data, batch_size=128, input_nodes=None, num_neighbors=[20, 10], **kwargs):
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
            # ensure batch attribute exists
            if not hasattr(batch, 'batch') or batch.batch is None:
                batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)

            yield {'pos': batch}

    def __len__(self):
        return len(self.loader)