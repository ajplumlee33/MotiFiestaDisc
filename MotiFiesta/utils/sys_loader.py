from torch_geometric.loader import NeighborLoader
import torch

def fast_vectorized_swap(edge_index, swap_ratio=0.5):
    """
    native PyTorch degree-preserving swap
    preserves 'source' node degrees (row)
    """
    # edge_index is shape [2, E]
    row = edge_index[0]
    col = edge_index[1]
    num_edges = row.size(0)
    
    mask = torch.rand(num_edges, device=edge_index.device) < swap_ratio
    swap_idx = torch.where(mask)[0]
    
    if len(swap_idx) > 1:
        new_col = col.clone()
        # shuffle the 'target' nodes for the selected edges
        shuffled_indices = torch.randperm(len(swap_idx), device=edge_index.device)
        new_col[swap_idx] = col[swap_idx][shuffled_indices]
        return torch.stack([row, new_col], dim=0)
    
    return edge_index

class SysLoader:
    def __init__(self, data, batch_size=128, input_nodes=None, num_neighbors=[20, 10], swap_ratio=0.5, **kwargs):
        self.swap_ratio = swap_ratio
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
            # if NeighborLoader didn't create a batch vector, create one of zeros
            if not hasattr(batch, 'batch') or batch.batch is None:
                batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)
            
            # 'positive' batch (real PPI neighborhood)
            pos_batch = batch
            
            # 'negative' batch (null model)
            neg_batch = pos_batch.clone()
            neg_batch.edge_index = fast_vectorized_swap(
                pos_batch.edge_index, 
                swap_ratio=self.swap_ratio
            )
            
            # yield as a dictionary to stay compatible with sys_train/motif_train
            yield {'pos': pos_batch, 'neg': neg_batch}

    def __len__(self):
        return len(self.loader)