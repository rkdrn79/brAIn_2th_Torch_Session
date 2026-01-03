import torch

outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
labels = torch.tensor([1, 0])

preds = torch.argmax(outputs, dim=2)

accuracy = (preds == labels).float().mean()
print("Accuracy:", accuracy.item())