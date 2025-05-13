import pandas as pd
from src.dataset.dataset import CustomDataset
from src.dataset.data_collator import CustomDataCollator
from torchvision import transforms
from torchvision.datasets import MNIST
import torch

def get_dataset(args):
    """
    Function to get the dataset based on the provided arguments.
    
    Args:
        args: Command line arguments containing data directory and other configurations.
    
    Returns:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        data_collator: The data collator for batching.
    """
    # Load MNIST dataset

    # 데이터 전처리 정의
    transform = transforms.ToTensor()

    # 학습용 데이터셋 다운로드 및 로드
    train_dataset = MNIST(root=args.data_dir, train=True, download=True, transform=transform)

    # 테스트용 데이터셋 다운로드 및 로드
    valid_dataset = MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    # DataLoader로 배치화
    train_dataset = CustomDataset(args, train_dataset)
    val_dataset = CustomDataset(args, valid_dataset)

    # Data collator
    data_collator = CustomDataCollator(args)

    return train_dataset, val_dataset, data_collator