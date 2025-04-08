import torch

x = torch.randn(10, 5)
w = torch.randn(3, 2) 

y = x @ w 
print(y)