import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_dataset = datasets.FakeData(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3*224*224, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(3):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")


"""
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

def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*224*224, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def train_model(model, train_loader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()}")

def main():
    train_loader = load_data()
    model = create_model()
    train_model(model, train_loader)

if __name__ == "__main__":
    main()
"""