import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, image_size=28):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)  # 유지: [B, hidden_dim, 28, 28]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                         # → [B, hidden_dim, 14, 14]
        self.relu = nn.ReLU()
        
        # fc 입력 크기 계산: hidden_dim * (image_size // 2) * (image_size // 2)
        flattened_dim = hidden_dim * (image_size // 2) * (image_size // 2)
        self.fc1 = nn.Linear(flattened_dim, output_dim)

    def forward(self, x):
        x = self.conv1(x)  # [B, hidden_dim, 28, 28]
        x = self.relu(x)
        x = self.pool(x)   # [B, hidden_dim, 14, 14]
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x