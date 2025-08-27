"""
Unified Training Engine for Gamma Hedge System

This module consolidates the functionality from both Trainer and ProductionTrainer
into a single, simplified training engine that follows the new architecture principles.

Replaces:
- training/trainer.py (legacy Trainer class)
- training/production_trainer.py (legacy ProductionTrainer class)

Key improvements:
- Single training interface for all scenarios
- Integrated with new configuration system
- Uses unified data interfaces
- Simplified error handling and monitoring
- Clean separation of concerns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from core.config import Config
from core.interfaces import (
    TrainingData, DataProvider, TrainingPhase, 
    TrainingEngine, BaseTrainer
)
from utils.logger import get_logger
from models.policy_network import PolicyNetwork
from utils.policy_tracker import global_policy_tracker

logger = get_logger(__name__)


class UnifiedTrainingEngine(BaseTrainer, TrainingEngine):
    """
    Unified training engine that handles all training scenarios
    
    This class consolidates the functionality of both the original Trainer
    and ProductionTrainer classes into a single, clean interface that
    integrates with the new modular architecture.
    """
    
    def __init__(self,
                 config: Config,
                 data_provider: DataProvider,
                 device: Optional[str] = None,
                 checkpoint_dir: str = "checkpoints",
                 enable_tracking: bool = True):
        
        # Initialize base trainer
        super().__init__(device or config.training.device)
        
        self.config = config
        self.data_provider = data_provider
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_tracking = enable_tracking
        global_policy_tracker.enable_tracking = enable_tracking
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        # Progress tracking
        self.progress_callback: Optional[Callable] = None
        
        logger.info(f"Unified training engine initialized with device: {self.device}")
    
    def _create_model(self) -> PolicyNetwork:
        """Create policy network based on configuration"""
        # Calculate input dimensions
        data_info = self.data_provider.get_data_info()
        
        if self.config.delta_hedge.enable_delta:
            n_assets = 1  # Delta hedging uses single underlying
        else:
            n_assets = self.config.market.n_assets
        
        # Calculate input dimension using centralized dimension management
        from core.dimensions import get_dimension_calculator
        
        # Get dimension calculator and compute input dimensions
        dim_calculator = get_dimension_calculator(n_assets)
        dim_spec = dim_calculator.calculate_model_input_dim(
            include_history=True,
            include_time=True,
            include_delta=True
        )
        
        input_dim = dim_spec.total
        
        logger.info(f"Model input dimensions calculated: {dim_spec}")
        logger.debug(f"Dimension breakdown: {dim_spec.to_dict()}")
        
        model = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=self.config.model.hidden_dims,
            dropout_rate=self.config.model.dropout_rate
        )
        
        logger.info(f"Created policy network: input_dim={input_dim}, "
                   f"hidden_dims={self.config.model.hidden_dims}")
        
        return model
    
    def _setup_training_components(self):
        """Setup model, optimizer, and scheduler"""
        if self.model is None:
            self.model = self._create_model()
            self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=1e-5
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=10,
            factor=0.5,
            verbose=True
        )
        
        logger.info("Training components setup completed")
    
    def _compute_policy_loss(self, 
                           batch: TrainingData,
                           actions: torch.Tensor,
                           returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute policy gradient loss with regularization
        
        Args:
            batch: Training data batch
            actions: Policy actions (execution probabilities)
            returns: Computed returns/rewards
            
        Returns:
            Dictionary with loss components
        """
        # REINFORCE policy gradient loss
        log_probs = torch.log(actions + 1e-8)  # Add small epsilon for numerical stability
        policy_loss = -(log_probs * returns).mean()
        
        # Entropy regularization to encourage exploration
        entropy = -(actions * torch.log(actions + 1e-8)).mean()
        entropy_loss = -self.config.training.entropy_weight * entropy
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'entropy': entropy
        }
    
    def _compute_returns(self, batch: TrainingData) -> torch.Tensor:
        """
        Compute returns (rewards) for the batch
        
        Args:
            batch: Training data batch
            
        Returns:
            Computed returns tensor
        """
        batch_size, seq_len = batch.prices.shape[:2]
        
        # Simple cost-based reward
        # In practice, this would be more sophisticated
        position_changes = torch.diff(batch.positions, dim=1)
        price_changes = torch.diff(batch.prices, dim=1)
        
        # Market impact cost (simplified)
        impact_costs = torch.abs(position_changes) * self.config.training.base_execution_cost
        
        # Price movement cost
        movement_costs = torch.abs(position_changes * price_changes) * self.config.training.variable_execution_cost
        
        # Total costs per step
        step_costs = impact_costs + movement_costs
        
        # Sum over sequence and assets, negate for reward
        returns = -step_costs.sum(dim=(1, 2))
        
        return returns
    
    def train_step(self, batch: TrainingData) -> Dict[str, float]:
        """Execute single training step"""
        self.model.train()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Create model input: [prev_holding, current_holding, prices, time_features]
        # Simplified for now - in practice would be more sophisticated
        batch_size, seq_len, n_assets = batch.prices.shape
        
        # Create time features (time remaining)
        time_features = torch.linspace(1.0, 0.0, seq_len).unsqueeze(0).unsqueeze(-1)
        time_features = time_features.expand(batch_size, seq_len, n_assets).to(self.device)
        
        # Previous holdings (shift current holdings by one step)
        prev_holdings = torch.cat([
            torch.zeros(batch_size, 1, n_assets, device=self.device),
            batch.positions[:, :-1, :]
        ], dim=1)
        
        # Combine features
        model_input = torch.cat([
            prev_holdings,
            batch.positions,
            batch.prices,
            time_features
        ], dim=-1)
        
        # Reshape for model: (batch_size * seq_len, input_dim)
        model_input = model_input.view(-1, model_input.shape[-1])
        
        # Forward pass
        actions = self.model(model_input)  # Shape: (batch_size * seq_len, 1)
        actions = actions.view(batch_size, seq_len, -1)  # Reshape back
        
        # Compute returns
        returns = self._compute_returns(batch)
        
        # Expand returns for sequence length
        returns_expanded = returns.unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)
        
        # Compute loss
        loss_dict = self._compute_policy_loss(batch, actions, returns_expanded)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Convert to float for logging
        metrics = {key: value.item() for key, value in loss_dict.items()}
        metrics['returns_mean'] = returns.mean().item()
        metrics['returns_std'] = returns.std().item()
        
        return metrics
    
    def validation_step(self, batch: TrainingData) -> Dict[str, float]:
        """Execute single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = batch.to(self.device)
            
            # Similar to train_step but without gradient computation
            batch_size, seq_len, n_assets = batch.prices.shape
            
            # Create time features
            time_features = torch.linspace(1.0, 0.0, seq_len).unsqueeze(0).unsqueeze(-1)
            time_features = time_features.expand(batch_size, seq_len, n_assets).to(self.device)
            
            # Previous holdings
            prev_holdings = torch.cat([
                torch.zeros(batch_size, 1, n_assets, device=self.device),
                batch.positions[:, :-1, :]
            ], dim=1)
            
            # Combine features
            model_input = torch.cat([
                prev_holdings,
                batch.positions,
                batch.prices,
                time_features
            ], dim=-1)
            
            # Reshape and forward pass
            model_input = model_input.view(-1, model_input.shape[-1])
            actions = self.model(model_input)
            actions = actions.view(batch_size, seq_len, -1)
            
            # Compute returns and loss
            returns = self._compute_returns(batch)
            returns_expanded = returns.unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)
            loss_dict = self._compute_policy_loss(batch, actions, returns_expanded)
            
            # Convert to float
            metrics = {key: value.item() for key, value in loss_dict.items()}
            metrics['returns_mean'] = returns.mean().item()
            
            return metrics
    
    def train(self, 
             n_epochs: int,
             batch_size: int,
             validation_data: Optional[DataProvider] = None,
             resume: bool = False,
             checkpoint_interval: int = 10,
             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train model with provided data
        
        Args:
            n_epochs: Number of training epochs
            batch_size: Training batch size
            validation_data: Optional validation data provider
            resume: Whether to resume from checkpoint
            checkpoint_interval: Save checkpoint every N epochs
            progress_callback: Optional progress callback function
            
        Returns:
            Training results and history
        """
        self.progress_callback = progress_callback
        
        # Setup training components
        self._setup_training_components()
        
        # Resume from checkpoint if requested
        if resume:
            self._load_latest_checkpoint()
        
        logger.info(f"Starting training for {n_epochs} epochs")
        
        for epoch in range(self.start_epoch, n_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(batch_size)
            
            # Validation phase
            val_metrics = {}
            if validation_data:
                val_metrics = self._validate_epoch(validation_data, batch_size)
            
            # Update learning rate scheduler
            if val_metrics and 'total_loss' in val_metrics:
                self.scheduler.step(val_metrics['total_loss'])
            else:
                self.scheduler.step(train_metrics['total_loss'])
            
            # Track training history
            epoch_time = time.time() - epoch_start_time
            epoch_metrics = {
                'epoch': epoch,
                'train': train_metrics,
                'validation': val_metrics,
                'epoch_time': epoch_time,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            self.training_history.append(epoch_metrics)
            
            # Check for best model
            current_loss = val_metrics.get('total_loss', train_metrics['total_loss'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch
                self._save_best_model()
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(epoch)
            
            # Progress callback
            if self.progress_callback:
                self.progress_callback({
                    'epoch': epoch,
                    'total_epochs': n_epochs,
                    'metrics': epoch_metrics,
                    'best_epoch': self.best_epoch,
                    'best_loss': self.best_loss
                })
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{n_epochs}: "
                       f"train_loss={train_metrics['total_loss']:.4f}, "
                       f"val_loss={val_metrics.get('total_loss', 'N/A')}, "
                       f"time={epoch_time:.1f}s")
        
        # Save final model
        self._save_checkpoint(n_epochs - 1, final=True)
        
        results = {
            'history': self.training_history,
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'final_metrics': self.training_history[-1] if self.training_history else {},
            'model_path': str(self.checkpoint_dir / 'best_model.pth'),
            'checkpoints': self._list_checkpoints()
        }
        
        logger.info(f"Training completed. Best epoch: {self.best_epoch}, Best loss: {self.best_loss:.4f}")
        
        return results
    
    def _train_epoch(self, batch_size: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Get training data
        train_data = self.data_provider.get_training_data(
            batch_size=batch_size,
            sequence_length=self.config.delta_hedge.sequence_length,
            phase=TrainingPhase.TRAIN
        )
        
        # Single batch training for now - could be extended to multiple batches
        metrics = self.train_step(train_data)
        
        return metrics
    
    def _validate_epoch(self, validation_data: DataProvider, batch_size: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        # Get validation data
        val_data = validation_data.get_training_data(
            batch_size=batch_size,
            sequence_length=self.config.delta_hedge.sequence_length,
            phase=TrainingPhase.VALIDATION
        )
        
        # Single batch validation
        metrics = self.validation_step(val_data)
        
        return metrics
    
    def evaluate(self, n_samples: int = 1000, batch_size: int = 32) -> Dict[str, Any]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not initialized. Train first or load checkpoint.")
        
        self.model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for _ in range(n_samples // batch_size):
                # Get evaluation data
                eval_data = self.data_provider.get_training_data(
                    batch_size=batch_size,
                    sequence_length=self.config.delta_hedge.sequence_length,
                    phase=TrainingPhase.TEST
                )
                
                metrics = self.validation_step(eval_data)
                all_metrics.append(metrics)
        
        # Aggregate metrics
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            return avg_metrics
        else:
            return {}
    
    def _save_checkpoint(self, epoch: int, final: bool = False):
        """Save training checkpoint"""
        if final:
            filename = "final_model.pth"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
        
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, filepath)
        
        # Update latest checkpoint link
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        if latest_path.exists():
            latest_path.unlink()
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Saved checkpoint: {filename}")
    
    def _save_best_model(self):
        """Save best model checkpoint"""
        if self.model is None:
            return
        
        filepath = self.checkpoint_dir / "best_model.pth"
        
        checkpoint = {
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'best_loss': self.best_loss,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved best model (epoch {self.best_epoch}, loss {self.best_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Setup model if not already done
        if self.model is None:
            self._setup_training_components()
        
        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.training_history = checkpoint.get('training_history', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint.get('metadata', {})
    
    def _load_latest_checkpoint(self):
        """Load the latest checkpoint if available"""
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        if latest_path.exists():
            self.load_checkpoint(str(latest_path))
            logger.info("Resumed from latest checkpoint")
        else:
            logger.info("No checkpoint found, starting from scratch")
    
    def _list_checkpoints(self) -> List[str]:
        """List available checkpoint files"""
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pth"):
            checkpoints.append(str(checkpoint_file))
        return sorted(checkpoints)


# Convenience functions for backward compatibility

def create_training_engine(config: Config, data_provider: DataProvider, **kwargs) -> UnifiedTrainingEngine:
    """Create unified training engine instance"""
    return UnifiedTrainingEngine(config, data_provider, **kwargs)