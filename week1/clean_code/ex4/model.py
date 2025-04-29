import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*224*224, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )