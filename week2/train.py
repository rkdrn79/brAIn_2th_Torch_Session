import os
import sys
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch

from arguments import get_arguments

from src.trainer import Trainer
from src.dataset.get_dataset import get_dataset
from src.model.get_model import get_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    set_seed(42)

    ## =================== Data =================== ##
    train_dataset, val_dataset, data_collator = get_dataset(args)

    num_workers = min(4, os.cpu_count())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.per_device_eval_batch_size, 
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )

    ## =================== Model =================== ##
    model = get_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ## =================== Training =================== ##
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        checkpoint_dir=f"./ckpt/{args.save_dir}",
        args=args
    )
    
    print(args)
    
    trainer.train()


if __name__=="__main__":

    args = get_arguments()
    main(args)