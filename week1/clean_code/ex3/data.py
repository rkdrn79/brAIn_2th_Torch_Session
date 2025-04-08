# data.py
import torch

def generate_linear_data(n_samples=100):
    x = torch.randn(n_samples, 1)
    y = 3 * x + 1 + 0.1 * torch.randn(n_samples, 1)
    return x, y