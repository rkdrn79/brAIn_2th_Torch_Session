import torch 
import torch.nn as nn

class CustomDataCollator:
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        # Unzip the batch
        data, labels = zip(*batch)
        
        # Convert to tensors
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)


        return {
            'inputs' : data,
            'targets' : labels
        }