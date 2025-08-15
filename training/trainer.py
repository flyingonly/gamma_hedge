import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
import numpy as np
import os
from datetime import datetime

class Trainer:
    """Trainer for policy network"""
    
    def __init__(self,
                 policy_network: nn.Module,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: str = 'checkpoints'):
        
        self.policy = policy_network.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_function = None  # Will be set during training
        self.checkpoint_dir = checkpoint_dir
        self.start_epoch = 0
        self.history = []
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def save_checkpoint(self, epoch: int, history: List[Dict], 
                       filename: Optional[str] = None) -> str:
        """Save training checkpoint"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'checkpoint_epoch_{epoch}_{timestamp}.pth'
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': history,
            'device': self.device
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        # Also save as latest checkpoint for easy resume
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        return filepath
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict:
        """Load training checkpoint"""
        if checkpoint_path is None:
            # Try to load latest checkpoint
            latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_path):
                checkpoint_path = latest_path
            else:
                print("No checkpoint found to load")
                return {}
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            return {}
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        self.history = checkpoint.get('history', [])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"Resumed from epoch {self.start_epoch}")
        return checkpoint
        
    def compute_policy_loss(self,
                           prices: torch.Tensor,
                           holdings: torch.Tensor) -> torch.Tensor:
        """
        Compute policy loss using REINFORCE-style gradient estimation
        """
        batch_size, n_timesteps, n_assets = prices.shape
        
        # Initial holding (from data)
        initial_holding = holdings[:, 0, :]
        current_holding = initial_holding.clone()
        
        total_loss = torch.zeros(1, device=self.device)
        
        for t in range(n_timesteps):
            current_price = prices[:, t, :]
            required_holding = holdings[:, t, :]
            
            # Get policy output
            exec_prob = self.policy(
                current_holding,
                required_holding,
                current_price
            )
            
            # Compute advantage (simplified - could use baseline)
            holding_change = required_holding - current_holding
            immediate_cost = -(holding_change * current_price).sum(dim=-1, keepdim=True)
            
            # Policy gradient loss
            loss = -exec_prob * immediate_cost.detach()
            total_loss = total_loss + loss.mean()
            
            # Update holdings (using exec_prob for differentiability)
            current_holding = exec_prob * required_holding + \
                            (1 - exec_prob) * current_holding
        
        return total_loss / n_timesteps
    
    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.policy.train()
        epoch_losses = []
        
        for prices, holdings in tqdm(data_loader, desc="Training"):
            prices = prices.to(self.device)
            holdings = holdings.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self.compute_policy_loss(prices, holdings)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        return {
            'loss': np.mean(epoch_losses),
            'loss_std': np.std(epoch_losses)
        }
    
    def train(self, 
              data_loader: DataLoader,
              n_epochs: int,
              val_loader: DataLoader = None,
              checkpoint_interval: int = 10,
              resume: bool = True) -> List[Dict[str, float]]:
        """
        Full training loop with checkpoint support
        
        Args:
            data_loader: Training data loader
            n_epochs: Number of epochs to train
            val_loader: Validation data loader (optional)
            checkpoint_interval: Save checkpoint every N epochs
            resume: Whether to resume from latest checkpoint
        """
        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()
        
        # Continue from where we left off
        history = self.history.copy()
        
        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            print(f"\nEpoch {epoch + 1}/{self.start_epoch + n_epochs}")
            
            # Training
            train_metrics = self.train_epoch(data_loader)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                train_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            history.append(train_metrics)
            
            # Print metrics
            print(f"Loss: {train_metrics['loss']:.4f} ± {train_metrics['loss_std']:.4f}")
            if val_loader is not None:
                print(f"Val Loss: {train_metrics.get('val_loss', 0):.4f}")
            
            # Save checkpoint at intervals
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch, history)
            
            # Save best model based on validation loss
            if val_loader is not None and len(history) > 0:
                current_val_loss = train_metrics.get('val_loss', float('inf'))
                best_val_loss = min([h.get('val_loss', float('inf')) for h in history])
                if current_val_loss <= best_val_loss:
                    self.save_checkpoint(epoch, history, 'best_model.pth')
                    print(f"New best model saved (val_loss: {current_val_loss:.4f})")
        
        # Save final checkpoint
        self.save_checkpoint(self.start_epoch + n_epochs - 1, history, 'final_checkpoint.pth')
        self.history = history
        
        return history
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set"""
        self.policy.eval()
        losses = []
        
        with torch.no_grad():
            for prices, holdings in data_loader:
                prices = prices.to(self.device)
                holdings = holdings.to(self.device)
                
                loss = self.compute_policy_loss(prices, holdings)
                losses.append(loss.item())
        
        return {
            'loss': np.mean(losses),
            'loss_std': np.std(losses)
        }