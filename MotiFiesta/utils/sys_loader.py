from torch_geometric.loader import NeighborLoader

import torch

def fast_vectorized_swap(edge_index, swap_ratio=0.5):
    """
    native PyTorch degree-preserving swap
    preserves 'source' node degrees
    """
    row, col = edge_index
    num_edges = row.size(0)
    
    # identify which edges to scramble (e.g., 50% of them)
    mask = torch.rand(num_edges, device=edge_index.device) < swap_ratio
    swap_idx = torch.where(mask)[0]
    
    # shuffle only the 'target' nodes (col) for the selected edges
    # keeps the 'source' (row) connectivity count the same
    new_col = col.clone()
    new_col[swap_idx] = col[swap_idx][torch.randperm(len(swap_idx), device=edge_index.device)]
    
    return torch.stack([row, new_col], dim=0)

class SysLoader:
    def __init__(self, data, batch_size=128, num_neighbors=[20, 10], swap_ratio=0.5, **kwargs):
        """
        wrapper for neighborloader that generates a null-model batch on the fly
        """
        self.swap_ratio = swap_ratio
        # initialize the standard PyG neighborloader
        self.loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            **kwargs
        )

    def __iter__(self):
        for batch in self.loader:
            # 'positive' batch (real PPI neighborhood)
            pos_batch = batch
            
            # 'negative' batch (null model)
            # clone the batch to keep features but swap the edges
            neg_batch = pos_batch.clone()
            neg_batch.edge_index = fast_vectorized_swap(
                pos_batch.edge_index, 
                swap_ratio=self.swap_ratio
            )
            
            # yield as a dictionary to stay compatible with motif_train
            yield {'pos': pos_batch, 'neg': neg_batch}

    def __len__(self):
        return len(self.loader)