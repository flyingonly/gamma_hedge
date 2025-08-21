import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
import numpy as np
import os
from datetime import datetime

# Import new interfaces (no sys.path needed)
from common.interfaces import DataResult
from common.adapters import to_data_result
from data.data_types import MarketData
from utils.policy_tracker import global_policy_tracker
from training.config import TrainingConfig
from models.value_function_network import ValueFunctionNetwork

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
        self.market_impact_scale = getattr(training_config, 'market_impact_scale', 0.01)
        
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
        
    def compute_policy_loss(self,
                           data: DataResult) -> torch.Tensor:
        """
        Compute policy loss using REINFORCE algorithm with cumulative cost.
        The reward is the negative of the total cost over the entire trajectory.
        """
        prices = data.prices
        holdings = data.holdings
        batch_size, n_timesteps, n_assets = prices.shape

        # Initial holding from data
        current_holding = holdings[:, 0, :].clone()

        log_probs = []
        rewards = []
        execution_probs = []  # Collect for entropy calculation

        # Start new episode for tracking (only track first batch item to avoid noise)
        if self.enable_tracking:
            global_policy_tracker.start_new_episode({
                'batch_size': batch_size,
                'n_timesteps': n_timesteps,
                'n_assets': n_assets
            })

        # --- Simulation Step ---
        # Simulate the policy over the entire sequence to collect trajectories
        for t in range(n_timesteps):
            current_price = prices[:, t, :]
            required_holding = holdings[:, t, :]
            
            # Create time features (time_remaining normalized between 0 and 1)
            time_remaining = torch.full((batch_size, 1), (n_timesteps - t - 1) / n_timesteps, 
                                      device=prices.device, dtype=prices.dtype)
            
            # Use time features if available in data, otherwise use computed time_remaining
            time_features = data.time_features[:, t, :] if data.time_features is not None else time_remaining

            # Get policy output (execution probability)
            exec_prob = self.policy(
                current_holding,
                required_holding,
                current_price,
                time_features=time_features
            )
            
            # Store execution probability for entropy calculation
            execution_probs.append(exec_prob)

            # Sample action and store log probability
            # Use a Bernoulli distribution for discrete action sampling
            dist = torch.distributions.Bernoulli(exec_prob)
            action = dist.sample()  # action=1 means execute, 0 means wait
            log_probs.append(dist.log_prob(action))

            # Calculate immediate cost for this step if action is taken
            holding_change = required_holding - current_holding
            market_impact_cost = -(holding_change * current_price).sum(dim=-1, keepdim=True)
            
            # Add execution cost penalty to encourage selective execution
            execution_cost = (self.base_execution_cost + 
                            self.variable_execution_cost * torch.abs(holding_change).sum(dim=-1, keepdim=True))
            immediate_cost = market_impact_cost + execution_cost
            
            # The cost is only incurred if the action is to execute
            cost_t = action * immediate_cost
            rewards.append(-cost_t) # Reward is the negative of the cost

            # Record decision for visualization (only first batch item)
            if self.enable_tracking and batch_size > 0:
                global_policy_tracker.record_decision(
                    timestep=t,
                    execution_probability=exec_prob[0, 0].cpu().item(),
                    action_taken=bool(action[0, 0].cpu().item()),
                    current_price=current_price[0, :],
                    current_holding=current_holding[0, :],
                    required_holding=required_holding[0, :],
                    immediate_cost=cost_t[0, 0].cpu().item()
                )

            # Update holdings based on the sampled action
            current_holding = action * required_holding + (1 - action) * current_holding

        # --- Terminal Forced Execution Step ---
        # Add mandatory terminal execution cost to align with value_function.py logic
        final_price = prices[:, -1, :]
        final_required_holding = holdings[:, -1, :]
        final_holding_change = final_required_holding - current_holding
        final_cost = -(final_holding_change * final_price).sum(dim=-1, keepdim=True)
        rewards.append(-final_cost)  # Add terminal cost as reward (negative cost)

        # Finalize episode tracking
        if self.enable_tracking:
            global_policy_tracker.finalize_episode()

        # --- Loss Calculation Step ---
        # Stack collected data
        log_probs = torch.stack(log_probs, dim=1)
        rewards = torch.stack(rewards, dim=1)
        execution_probs = torch.stack(execution_probs, dim=1)

        # Calculate the cumulative reward (G_t) for the whole trajectory
        # In this simplified REINFORCE, we use the total reward for every step
        total_reward = torch.sum(rewards, dim=1, keepdim=True)

        # Policy gradient loss: - (log_prob * total_reward)
        # We use .detach() on total_reward as per REINFORCE algorithm
        policy_loss = -(log_probs * total_reward.detach()).mean()
        
        # Add entropy regularization to encourage exploration
        # Entropy for Bernoulli distribution: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-8  # Small constant to avoid log(0)
        entropy = -(execution_probs * torch.log(execution_probs + eps) + 
                   (1 - execution_probs) * torch.log(1 - execution_probs + eps))
        entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy
        
        # Combine losses
        total_loss = policy_loss + self.entropy_weight * entropy_loss

        return total_loss
    
    def compute_policy_loss_with_advantage(self,
                                         data: DataResult) -> Dict[str, torch.Tensor]:
        """
        Improved policy loss computation using returns-to-go and value function baseline.
        Implements advantage function A(s,a) = returns_to_go - V(s) for reduced variance.
        """
        prices = data.prices
        holdings = data.holdings
        batch_size, n_timesteps, n_assets = prices.shape
        
        # Initialize tracking variables
        log_probs = []
        rewards = []
        execution_probs = []
        states = []  # For value function input
        
        # Initial holdings - start with first required holding (realistic trading scenario)
        current_holding = holdings[:, 0, :].clone()  # Start with first timestep's required holding
        
        # Start new episode for tracking (only track first batch item to avoid noise)
        if self.enable_tracking:
            global_policy_tracker.start_new_episode({
                'batch_size': batch_size,
                'n_timesteps': n_timesteps,
                'n_assets': n_assets
            })
        
        # --- Forward Pass: Collect trajectories ---
        for t in range(n_timesteps):
            current_price = prices[:, t, :]
            required_holding = holdings[:, t, :]
            
            # Compute remaining time
            time_remaining = torch.full((batch_size, 1), (n_timesteps - t - 1) / n_timesteps, device=self.device)
            
            # Use time features if available, otherwise use computed time_remaining
            time_features = data.time_features[:, t, :] if data.time_features is not None else time_remaining
            
            # Store state for value function
            state_input = {
                'prev_holding': current_holding,
                'current_holding': required_holding,
                'price': current_price,
                'time_features': time_features
            }
            states.append(state_input)
            
            # Get policy output (execution probability)
            exec_prob = self.policy(
                current_holding,
                required_holding,
                current_price,
                time_features=time_features
            )
            execution_probs.append(exec_prob)
            
            # Sample action and store log probability
            dist = torch.distributions.Bernoulli(exec_prob)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            # Calculate immediate cost
            holding_change = required_holding - current_holding
            
            # For t=0, we already hold the required position, so no real execution needed
            if t == 0:
                # At t=0, we already have the required holding, so no cost
                immediate_cost = torch.zeros(batch_size, 1, device=self.device)
            else:
                # Scale market impact to control reward magnitude
                market_impact_cost = -(holding_change * current_price).sum(dim=-1, keepdim=True) * self.market_impact_scale
                execution_cost = (self.base_execution_cost + 
                                self.variable_execution_cost * torch.abs(holding_change).sum(dim=-1, keepdim=True))
                immediate_cost = market_impact_cost + execution_cost
            
            # Cost only incurred if action is to execute
            cost_t = action * immediate_cost
            rewards.append(-cost_t)  # Reward is negative cost
            
            # Record decision for visualization (only first batch item)
            if self.enable_tracking and batch_size > 0:
                global_policy_tracker.record_decision(
                    timestep=t,
                    execution_probability=exec_prob[0, 0].cpu().item(),
                    action_taken=bool(action[0, 0].cpu().item()),
                    current_price=current_price[0, :],
                    current_holding=current_holding[0, :],
                    required_holding=required_holding[0, :],
                    immediate_cost=immediate_cost[0, 0].cpu().item()
                )
            
            # Update holdings
            current_holding = action * required_holding + (1 - action) * current_holding
        
        # --- Terminal Forced Execution (probability = 1, not affected by policy) ---
        final_price = prices[:, -1, :]
        final_required_holding = holdings[:, -1, :]
        final_holding_change = final_required_holding - current_holding
        # Terminal cost is ALWAYS incurred (probability = 1) - scale market impact
        final_cost = -(final_holding_change * final_price).sum(dim=-1, keepdim=True) * self.market_impact_scale
        rewards.append(-final_cost)
        
        # Finalize episode tracking
        if self.enable_tracking:
            global_policy_tracker.finalize_episode()
        
        # --- Returns-to-go Calculation ---
        log_probs = torch.stack(log_probs, dim=1)  # (batch, timesteps, 1)
        rewards = torch.stack(rewards, dim=1)      # (batch, timesteps+1, 1) - includes terminal
        execution_probs = torch.stack(execution_probs, dim=1)  # (batch, timesteps, 1)
        
        # Compute returns-to-go for each timestep
        returns_to_go = torch.zeros_like(rewards[:, :-1, :])  # Exclude terminal for policy steps
        
        # Calculate returns-to-go: G_t = sum(rewards[t:])
        for t in range(n_timesteps):
            returns_to_go[:, t, :] = torch.sum(rewards[:, t:], dim=1)
        
        # --- Value Function Predictions and Loss ---
        value_loss = torch.tensor(0.0, device=self.device)
        if self.use_value_function and self.value_function is not None:
            # Get value predictions for all states
            value_predictions = []
            for t in range(n_timesteps):
                state = states[t]
                value_pred = self.value_function(
                    state['prev_holding'],
                    state['current_holding'], 
                    state['price'],
                    time_features=state['time_features']
                )
                value_predictions.append(value_pred)
            
            value_predictions = torch.stack(value_predictions, dim=1)  # (batch, timesteps, 1)
            
            # Compute value function loss
            value_loss = self.value_function.compute_value_loss(
                value_predictions, returns_to_go
            )
            
            # Compute advantages: A(s,a) = returns_to_go - V(s)
            advantages = returns_to_go - value_predictions.detach()
        else:
            # Fallback: use returns-to-go directly (original REINFORCE)
            advantages = returns_to_go
        
        # --- Policy Loss with Advantages ---
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # --- Entropy Regularization ---
        eps = 1e-8
        entropy = -(execution_probs * torch.log(execution_probs + eps) + 
                   (1 - execution_probs) * torch.log(1 - execution_probs + eps))
        entropy_loss = -entropy.mean()
        
        # --- Combined Loss ---
        total_loss = policy_loss + self.entropy_weight * entropy_loss
        if self.use_value_function:
            value_weight = getattr(self.config, 'value_loss_weight', 0.5)
            total_loss += value_weight * value_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'mean_advantage': advantages.mean(),
            'mean_return': returns_to_go.mean()
        }
    
    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.policy.train()
        epoch_losses = []
        
        for batch_data in tqdm(data_loader, desc="Training"):
            # Convert to unified DataResult format
            data_result = to_data_result(batch_data)
            data_result = data_result.to(self.device)
            
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
            for batch_data in data_loader:
                # Convert to unified DataResult format
                data_result = to_data_result(batch_data)
                data_result = data_result.to(self.device)
                
                loss_dict = self.compute_policy_loss_with_advantage(data_result)
                losses.append(loss_dict['total_loss'].item())
        
        return {
            'loss': np.mean(losses),
            'loss_std': np.std(losses)
        }