import torch

class RunningStats:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.count = 0
        self.running_mean = 0
        self.running_var = 1

    def push(self, x):
        """
        updates running mean and variance using a momentum-based moving average.
        x: the loss or value from the current batch (tensor or float).
        """
        # ensure x is a float for calculation
        val = x.item() if torch.is_tensor(x) else x
        
        if self.count == 0:
            self.running_mean = val
            self.running_var = 1.0 # Initial guess
        else:
            # exponential moving average update
            self.running_mean = (self.momentum * self.running_mean) + (1 - self.momentum) * val
            # update variance based on squared difference
            diff_sq = (val - self.running_mean) ** 2
            self.running_var = (self.momentum * self.running_var) + (1 - self.momentum) * diff_sq
        
        self.count += 1

    def mean(self):
        return self.running_mean

    def std(self):
        # return sqrt of variance with a small epsilon for stability
        return torch.sqrt(torch.tensor(self.running_var) + 1e-8)