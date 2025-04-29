import torch
import torch.nn as nn

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        self.args = args
        dataset = self.preprocess(dataset)
        self.data = dataset.drop(columns=['international']).values
        self.labels = dataset['international'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def preprocess(self, dataset):
        # international column -> True False encoding
        dataset['international'] = dataset['international'].apply(lambda x: 1 if x == True else 0)
        return dataset