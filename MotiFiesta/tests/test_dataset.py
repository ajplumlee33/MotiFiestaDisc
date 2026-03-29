import torch
from MotiFiesta.utils.sys_txt import SysTxtDataset

def test_sys_txt_dataset():
    # initialization
    dataset = SysTxtDataset(root='data/mips_torch')
    
    # extraction - now returns a single Data object instead of a dict
    data = dataset[0] 
    
    print("\n--- graph statistics ---")
    print(f"nodes: {data.num_nodes}")
    print(f"edges: {data.num_edges}")
    
    # feature logic - check if the 25-dim features were assigned correctly
    assert data.x.shape[1] == 25, "feature dimension should be 25"
    # noise dimensions should not be empty
    assert not torch.all(data.x[:, 2:] == 0)
    
    # check the 'size' and 'cat' slots (0 and 1)
    print(f"sample node features (size/cat): {data.x[0, :2].tolist()}")
    
    # node consistency - verify the node names were saved
    assert hasattr(data, 'node_names'), "node_names attribute missing"
    assert len(data.node_names) == data.num_nodes, "node name count mismatch"
    
    print("\ndataset tests passed")

if __name__ == "__main__":
    test_sys_txt_dataset()