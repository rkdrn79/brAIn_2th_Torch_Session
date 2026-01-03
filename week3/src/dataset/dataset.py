import torch
import torch.nn as nn

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, data):
        self.args = args
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return {"image": image, "label": label}