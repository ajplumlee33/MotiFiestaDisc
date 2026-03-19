from torch_geometric.loader import NeighborLoader

class SysLevLoader(NeighborLoader):
    def __init__(self, real_data, null_data, **kwargs):
        # initialize base loader on the real graph
        super().__init__(real_data, **kwargs)
        self.null_data = null_data
        
        # create a secondary internal loader for the null graph
        self.null_sampler = NeighborLoader(
            null_data,
            num_neighbors=kwargs.get('num_neighbors'),
            batch_size=kwargs.get('batch_size'),
            shuffle=False # sampling is driven by the real loader's indices
        )

    def __iter__(self):
        for batch_pos in super().__iter__():
            # sample the exact same nodes from the rewired null graph
            # we use the input_id (the root nodes of the current batch)
            batch_neg = self.null_sampler.filter_data(batch_pos.input_id)
            
            # yield both for the training loop
            yield {'pos': batch_pos, 'neg': batch_neg}