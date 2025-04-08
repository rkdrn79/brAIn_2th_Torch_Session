import torch
import torch.nn as nn
import torch.optim as optim

x = torch.randn(100, 1)
y = 3 * x + 1 + 0.1 * torch.randn(100, 1)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Final weight:", model.weight.data)
print("Final bias:", model.bias.data)

"""
import torch
import torch.nn as nn
import torch.optim as optim

def generate_data():
    x = torch.randn(100, 1)
    y = 3 * x + 1 + 0.1 * torch.randn(100, 1)
    return x, y

def create_model():
    return nn.Linear(1, 1)

def train(model, x, y, epochs=100, lr=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model

def print_parameters(model):
    print("Final weight:", model.weight.data)
    print("Final bias:", model.bias.data)

# 실행
x, y = generate_data()
model = create_model()
model = train(model, x, y)
print_parameters(model)
"""