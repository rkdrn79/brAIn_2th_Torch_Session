import torch

x = torch.randn(10, 5)
w = torch.randn(5, 2) 

breakpoint()
y = x @ w 
print(y)