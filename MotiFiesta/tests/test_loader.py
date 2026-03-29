import torch
from torch_geometric.data import Data
from MotiFiesta.utils.sys_loader import SysLoader

def test_sys_loader_behavior():
    # dummy graph (100 nodes, random edges)
    num_nodes = 100
    edge_index = torch.randint(0, num_nodes, (2, 500))
    x = torch.randn(num_nodes, 25) # Your 25-dim features
    data = Data(x=x, edge_index=edge_index)

    # setting swap_ratio to 1.0 for the test to ensure maximum change
    batch_size = 10
    loader = SysLoader(data, batch_size=batch_size, num_neighbors=[5], swap_ratio=1.0)

    print(f"checking {len(loader)} batches...")

    for i, batch_dict in enumerate(loader):
        pos = batch_dict['pos']
        neg = batch_dict['neg']

        # shape integrity
        assert pos.num_nodes == neg.num_nodes, "node count mismatch"
        assert pos.edge_index.size(1) == neg.edge_index.size(1), "edge count mismatch"
        
        # feature parity - features must remain untouched so the GNN sees the same biological signal
        assert torch.equal(pos.x, neg.x), "features were accidentally modified"

        # degree preservation - 'row' (source) is never changed in swap function
        # out-degree of every node in the batch must be identical
        pos_out_degrees = torch.bincount(pos.edge_index[0], minlength=pos.num_nodes)
        neg_out_degrees = torch.bincount(neg.edge_index[0], minlength=neg.num_nodes)
        
        assert torch.equal(pos_out_degrees, neg_out_degrees), "source degrees were not preserved"

        # edge randomization - some edges should be different if swap_ratio > 0
        is_different = not torch.equal(pos.edge_index, neg.edge_index)
        
        if i == 0:
            print(f"batch {i} - nodes: {pos.num_nodes}, edges: {pos.num_edges}")
            print(f"degree distribution preserved.")
            print(f"edge randomization active: {is_different}")
        
        if i > 2: break # test a few batches

    print("\nSysLoader validation passed")

if __name__ == "__main__":
    test_sys_loader_behavior()