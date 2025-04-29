import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(batch_size=64):
    transform = transforms.ToTensor()
    train_dataset = datasets.FakeData(transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader