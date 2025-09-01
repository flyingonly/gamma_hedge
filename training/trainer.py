import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
import numpy as np
import os
import logging
from datetime import datetime

# Import new interfaces (no sys.path needed)
from core.interfaces import DataResult
from data.data_types import MarketData
from utils.policy_tracker import global_policy_tracker
from core.config import TrainingConfig
from models.value_function_network import ValueFunctionNetwork
from training.loss_calculator import create_advanced_loss_calculator, LossCalculator
from utils.logger import get_logger

logger = get_logger(__name__)

class Trainer:
    """Trainer for policy network"""
    
    def __init__(self,
                 policy_network: nn.Module,
                 training_config: Optional[TrainingConfig] = None,
                 device: Optional[str] = None,
                 checkpoint_dir: str = 'checkpoints',
                 enable_tracking: bool = False,
                 use_value_function: bool = True):
        
        # Initialize configuration
        if training_config is None:
            training_config = TrainingConfig()
        self.config = training_config
        
        # Setup device
        if device is None:
            device = training_config.device if hasattr(training_config, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = policy_network.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=training_config.learning_rate)
        
        # Initialize value function network if enabled
        self.use_value_function = use_value_function
        if use_value_function:
            # Create value function with same input dimension as policy network
            self.value_function = ValueFunctionNetwork(
                input_dim=policy_network.layers[0].in_features,  # Match policy network input
                hidden_dims=getattr(training_config, 'value_hidden_dims', [128, 64, 32]),
                dropout_rate=getattr(training_config, 'value_dropout_rate', 0.1)
            ).to(device)
            
            # Separate optimizer for value function
            self.value_optimizer = optim.Adam(
                self.value_function.parameters(), 
                lr=getattr(training_config, 'value_learning_rate', training_config.learning_rate)
            )
        else:
            self.value_function = None
            self.value_optimizer = None
        self.checkpoint_dir = checkpoint_dir
        self.start_epoch = 0
        self.history = []
        self.enable_tracking = enable_tracking
        
        # Regularization parameters from configuration
        self.entropy_weight = training_config.entropy_weight
        self.base_execution_cost = training_config.base_execution_cost
        self.variable_execution_cost = training_config.variable_execution_cost
        self.market_impact_scale = training_config.market_impact_scale
        
        # Initialize advanced loss calculator with all components from unified config
        loss_config = {
            'policy_loss': training_config.policy_loss_weight,
            'value_loss': training_config.value_loss_weight if use_value_function else 0.0,
            'entropy_loss': training_config.entropy_weight,
            'l2_regularization': training_config.l2_regularization_weight,
            'sharpness_regularizer': training_config.sharpness_regularizer,
            'entropy_gate_k': training_config.entropy_gate_k,
            'temporal_push': training_config.temporal_push,
            'temporal_smooth': training_config.temporal_smooth
        }
        
        self.loss_calculator = create_advanced_loss_calculator(
            policy_network=self.policy,
            value_network=self.value_function,
            enable_tracking=enable_tracking,
            device=device,
            market_impact_scale=training_config.market_impact_scale,
            base_execution_cost=training_config.base_execution_cost,
            variable_execution_cost=training_config.variable_execution_cost,
            config=loss_config
        )
        
        # Configure policy tracker
        global_policy_tracker.enable_tracking = enable_tracking
        
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        self.history = checkpoint.get('history', [])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"Resumed from epoch {self.start_epoch}")
        return checkpoint
        
    
    def compute_policy_loss_with_advantage(self,
                                         data: DataResult) -> Dict[str, torch.Tensor]:
        """
        Compute policy loss using modular loss calculator with proper mask handling.
        
        This method delegates to the LossCalculator for complete policy execution simulation,
        terminal cost calculation, advantage computation, and final loss calculation.
        All mask logic and holding updates are handled correctly by the modular components.
        """
        # Delegate to the modular loss calculator
        return self.loss_calculator.compute_policy_loss_with_advantage(data)
    
    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.policy.train()
        epoch_losses = []
        
        for batch_data in tqdm(data_loader, desc="Training"):
            # Use DataResult format directly
            data_result = batch_data.to(self.device)
            
            # Zero gradients for both networks
            self.optimizer.zero_grad()
            if self.use_value_function and self.value_optimizer is not None:
                self.value_optimizer.zero_grad()
            
            # Compute improved loss with advantages
            loss_dict = self.compute_policy_loss_with_advantage(data_result)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for both networks
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            if self.use_value_function and self.value_function is not None:
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), max_norm=1.0)
            
            # Update weights for both networks
            self.optimizer.step()
            if self.use_value_function and self.value_optimizer is not None:
                self.value_optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
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
        if self.use_value_function and self.value_function is not None:
            self.value_function.eval()
        losses = []
        
        with torch.no_grad():
            # Add progress bar for validation step
            for batch_data in tqdm(data_loader, desc="Validating", leave=False):
                # Use DataResult format directly
                data_result = batch_data.to(self.device)
                
                loss_dict = self.compute_policy_loss_with_advantage(data_result)
                losses.append(loss_dict['total_loss'].item())
        
        return {
            'loss': np.mean(losses),
            'loss_std': np.std(losses)
        }
    
