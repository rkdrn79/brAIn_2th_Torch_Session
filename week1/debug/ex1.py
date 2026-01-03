import torch

x = torch.randn(10, 5)
w = torch.randn(4, 2) 

y = x @ w 
print(y)