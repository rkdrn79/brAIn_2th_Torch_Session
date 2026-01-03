from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

class MyDataset(Dataset):
    def __init__(self):
        self.data = [[1, 2], [3, 4], [5, 6]]
        self.labels = [0, 1, 0]

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, labels

    def __len__(self):
        return len(self.data)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=False)
model = Model()

for x, y in loader:
    out = model(x)
    print(out)