import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(1, 1)
model.weight.requires_grad = False  # 버그

x = torch.tensor([[1.0]])
y = torch.tensor([[2.0]])

optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

for i in range(5):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    print(f"Step {i}: loss = {loss.item()}")