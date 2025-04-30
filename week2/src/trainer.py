import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 optimizer,
                 device,
                 loss_fn=None,
                 epochs=100,
                 eval_steps=500,
                 checkpoint_dir="./ckpt",
                 args=None):

        # training arguments
        self.args = args
        self.epochs = epochs
        self.eval_steps = eval_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.steps = 0
        
        # model and optimizer
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # loss function (default to BCE)
        self.loss_fn = nn.BCELoss() if loss_fn is None else loss_fn

        # data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logs = []

    def train(self):
        # Zero gradients at the beginning
        self.optimizer.zero_grad()
        
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            
            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

        # Log Plot
        steps = [log["step"] for log in self.train_logs]
        epochs = [log["epoch"] for log in self.train_logs]
        train_loss = [log["train_loss"] for log in self.train_logs]
        eval_loss = [log["eval_loss"] for log in self.train_logs]
        f1 = [log["f1"] for log in self.train_logs]
        precision = [log["precision"] for log in self.train_logs]
        recall = [log["recall"] for log in self.train_logs]
        accuracy = [log["accuracy"] for log in self.train_logs]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_loss, label="Train Loss")
        plt.plot(steps, eval_loss, label="Eval Loss")
        plt.plot(steps, f1, label="F1 Score")
        plt.plot(steps, precision, label="Precision")
        plt.plot(steps, recall, label="Recall")
        plt.plot(steps, accuracy, label="Accuracy")
        plt.xlabel("Steps")
        plt.ylabel("Metric")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_progress.png")
        plt.show()

                
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "final_model.pt"))
        print(f"Training completed. Final model saved to {os.path.join(self.checkpoint_dir, 'final_model.pt')}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        #for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
        for batch_idx, batch in enumerate(self.train_loader):
            # Compute loss
            loss, _ = self.compute_loss(batch)
            
            # Backward pass
            # TODO : backward pass

            total_loss += loss.item()

            self.steps += 1
            # Evaluate during training
            if self.steps % self.eval_steps == 0:
                eval_metrics = self.evaluate()
                self.model.train()  # Switch back to training mode
                
                fractional_epoch = epoch + (batch_idx + 1) / len(self.train_loader)
                print(f"Step {self.steps} - Epoch {fractional_epoch:.2f} - Training loss: {loss.item():.4f}, "
                    f"Evaluation loss: {eval_metrics['eval/loss']:.4f}, "
                    f"f1_score: {eval_metrics['eval/f1_score']:.4f}, "
                    f"Precision: {eval_metrics['eval/precision']:.4f}, "
                    f"Recall: {eval_metrics['eval/recall']:.4f}, "
                    f"Accuracy: {eval_metrics['eval/accuracy']:.4f}")
                
                log = {
                    "step": self.steps,
                    "epoch": fractional_epoch,
                    "train_loss": loss.item(),
                    "eval_loss": eval_metrics['eval/loss'],
                    "f1": eval_metrics['eval/f1_score'],
                    "precision": eval_metrics['eval/precision'],
                    "recall": eval_metrics['eval/recall'],
                    "accuracy": eval_metrics['eval/accuracy']
                }
                self.train_logs.append(log)

    
    def compute_loss(self, batch):
        # Extract inputs and targets from batch
        inputs = batch['inputs'].to(self.device)
        targets = batch['targets'].to(self.device)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets.unsqueeze(1))

        return loss, outputs

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            #for batch in tqdm(self.val_loader, desc="Evaluation"):
            for batch in self.val_loader:
                loss, outputs = self.compute_loss(batch)
                total_loss += loss.item()

                # Store predictions and targets for metrics calculation
                all_targets.append(batch['targets'].cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # Calculate average loss
        avg_loss = total_loss / len(self.val_loader)
        
        # Prepare data for metrics calculation
        y_true = np.concatenate(all_targets)
        y_pred_raw = np.concatenate(all_outputs)
        
        # Apply threshold for binary classification
        y_pred = (y_pred_raw > 0.5).astype(int)

        # Calculate metrics
        metrics = {
            "eval/loss": avg_loss,
            "eval/f1_score": f1_score(y_true, y_pred, average='macro'),
            "eval/precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "eval/recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "eval/accuracy": accuracy_score(y_true, y_pred)
        }
        
        return metrics

    def save_checkpoint(self, epoch):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {start_epoch})")
        return start_epoch