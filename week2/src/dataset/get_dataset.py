import pandas as pd
from src.dataset.dataset import CustomDataset
from src.dataset.data_collator import CustomDataCollator
from sklearn.model_selection import train_test_split

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
    # Load dataset
    train_data = pd.read_csv(args.data_dir + '/train.csv')

    train_data, val_data = train_test_split(train_data, test_size=args.val_ratio, random_state=42)
    args.input_dim = train_data.shape[1] - 1

    # Make Custom dataset
    train_dataset = CustomDataset(args, train_data)
    val_dataset = CustomDataset(args, val_data)

    # Data collator
    data_collator = CustomDataCollator(args)

    return train_dataset, val_dataset, data_collator