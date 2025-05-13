import argparse


def get_arguments():
    
    parser = argparse.ArgumentParser(description="MLP Training Script")
    #================= parser with data  ===========================#
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')

    #================= parser with model  ===========================#
    parser.add_argument('--model_type', type=str, default='CNN', help='Model type')
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=10, help='Output dimension')

    #================= parser with train  ===========================#    
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32, help='Per device train batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32, help='Per device eval batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluate per {eval_steps} steps')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save per {save_steps} steps')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')

    #================= parser with save, load  ===========================#
    parser.add_argument('--save_dir', type=str, default='baseline', help='Save directory')
    parser.add_argument('--load_dir', type=str, default='baseline', help='Load directory')

    args = parser.parse_args()
    return args

    args = parser.parse_args()
    
    return args