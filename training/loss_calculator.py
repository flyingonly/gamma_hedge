import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional
from core.interfaces import DataResult

class LossComponent(ABC):
    """Base class for modular loss components"""
    
    @abstractmethod
    def compute_loss(self, **kwargs) -> torch.Tensor:
        """Compute the loss component"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the loss component"""
        pass

class PolicyLossComponent(LossComponent):
    """Policy gradient loss component using advantage function"""
    
    def compute_loss(self, 
                    log_probs: torch.Tensor,
                    advantages: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """Compute policy gradient loss: -E[log π(a|s) * A(s,a)]"""
        return -(log_probs * advantages.detach()).mean()
    
    def get_name(self) -> str:
        return "policy_loss"

class ValueLossComponent(LossComponent):
    """Value function loss component"""
    
    def compute_loss(self,
                    value_predictions: torch.Tensor,
                    target_returns: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """Compute value function loss: MSE between V(s) and returns"""
        return nn.functional.mse_loss(value_predictions, target_returns.detach())
    
    def get_name(self) -> str:
        return "value_loss"

class EntropyLossComponent(LossComponent):
    """Entropy regularization loss component"""
    
    def compute_loss(self,
                    execution_probs: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """Compute entropy loss for Bernoulli distribution"""
        eps = 1e-8
        entropy = -(execution_probs * torch.log(execution_probs + eps) + 
                   (1 - execution_probs) * torch.log(1 - execution_probs + eps))
        return -entropy.mean()  # Negative because we want to maximize entropy
    
    def get_name(self) -> str:
        return "entropy_loss"

class L2RegularizationComponent(LossComponent):
    """L2 regularization loss component for policy parameters"""
    
    def __init__(self, policy_network: nn.Module):
        self.policy_network = policy_network
    
    def compute_loss(self, **kwargs) -> torch.Tensor:
        """Compute L2 regularization loss"""
        l2_loss = torch.tensor(0.0, device=next(self.policy_network.parameters()).device)
        for param in self.policy_network.parameters():
            l2_loss += torch.norm(param, p=2) ** 2
        return l2_loss
    
    def get_name(self) -> str:
        return "l2_regularization"

class PolicyExecutionComponent(LossComponent):
    """Complete policy execution simulation component with proper mask handling"""
    
    def __init__(self, policy_network, value_network=None, enable_tracking=False, device='cpu',
                 market_impact_scale=1.0, base_execution_cost=0.0, variable_execution_cost=0.0):
        self.policy_network = policy_network
        self.value_network = value_network
        self.enable_tracking = enable_tracking
        self.device = device
        self.market_impact_scale = market_impact_scale
        self.base_execution_cost = base_execution_cost
        self.variable_execution_cost = variable_execution_cost
    
    def compute_loss(self, data: 'DataResult', **kwargs) -> Dict[str, torch.Tensor]:
        """Execute complete policy simulation with proper mask logic"""
        from utils.policy_tracker import global_policy_tracker
        from utils.logger import get_logger
        logger = get_logger(__name__)
        
        prices = data.prices
        holdings = data.holdings
        
        # Handle variable-length collate output shape normalization
        if len(prices.shape) == 4:
            batch_size, n_sequences, n_timesteps, n_assets = prices.shape
            if n_sequences == 1:
                prices = prices.squeeze(1)
                holdings = holdings.squeeze(1)
            else:
                raise ValueError(f"Multiple sequences per batch not supported: {prices.shape}")
        
        if len(prices.shape) == 3:
            batch_size, n_timesteps, n_assets = prices.shape
        elif len(prices.shape) == 2:
            n_timesteps, n_assets = prices.shape
            batch_size = 1
            prices = prices.unsqueeze(0)
            holdings = holdings.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected tensor shape for prices: {prices.shape}")
        
        # Initialize tracking variables
        log_probs = []
        rewards = []
        execution_probs = []
        states = []  # For value function input
        
        # Initial holdings - start with first required holding
        current_holding = holdings[:, 0, :].clone()
        
        # Start tracking episode if enabled
        if self.enable_tracking:
            global_policy_tracker.start_new_episode({
                'batch_size': batch_size,
                'n_timesteps': n_timesteps,
                'n_assets': n_assets
            })
        
        # Get attention mask for variable-length sequences
        attention_mask = data.metadata.get('attention_mask')
        
        # Forward pass: collect trajectories with proper mask handling
        for t in range(n_timesteps):
            current_price = prices[:, t, :]
            required_holding = holdings[:, t, :]
            
            # Compute time features
            time_remaining = torch.full((batch_size, 1), (n_timesteps - t - 1) / n_timesteps, device=self.device)
            time_features = data.time_features[:, t, :] if data.time_features is not None else time_remaining
            history_features = data.history_features[:, t, :] if data.history_features is not None else None
            delta_features = data.delta_features[:, t, :] if data.delta_features is not None else None
            
            # Store state for value function
            state_input = {
                'prev_holding': current_holding,
                'current_holding': required_holding,
                'price': current_price,
                'time_features': time_features,
                'history_features': history_features,
                'delta_features': delta_features
            }
            states.append(state_input)
            
            # Get policy output (execution probability)
            exec_prob = self.policy_network(
                current_holding,
                required_holding,
                current_price,
                history=history_features,
                time_features=time_features,
                delta_features=delta_features
            )
            execution_probs.append(exec_prob)
            
            # Sample action and store log probability
            dist = torch.distributions.Bernoulli(exec_prob)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            # Calculate immediate cost with mask handling
            if attention_mask is not None:
                timestep_mask = attention_mask[:, t].unsqueeze(-1).float().to(current_price.device)
            else:
                timestep_mask = torch.ones(batch_size, 1, device=current_price.device)
            
            holding_change = required_holding - current_holding
            
            if t == 0:
                # No cost at initial timestep
                immediate_cost = torch.zeros(batch_size, 1, device=self.device)
            else:
                # Market impact and execution costs
                market_impact_cost = -(holding_change * current_price).sum(dim=-1, keepdim=True) * self.market_impact_scale
                execution_cost = (self.base_execution_cost + 
                                self.variable_execution_cost * torch.abs(holding_change).sum(dim=-1, keepdim=True))
                immediate_cost = market_impact_cost + execution_cost
            
            # Cost only incurred if action is to execute, masked for padded positions
            action = action * timestep_mask  # Zero out action for padded positions
            cost_t = action * immediate_cost
            rewards.append(-cost_t)  # Reward is negative cost
            
            # Record decision for tracking (only first batch item)
            if self.enable_tracking and batch_size > 0:
                is_valid_timestep = (attention_mask is None or attention_mask[0, t].item())
                if is_valid_timestep:
                    global_policy_tracker.record_decision(
                        timestep=t,
                        execution_probability=exec_prob[0, 0].cpu().item(),
                        action_taken=bool(action[0, 0].cpu().item()),
                        current_price=current_price[0, :],
                        current_holding=current_holding[0, :],
                        required_holding=required_holding[0, :],
                        immediate_cost=immediate_cost[0, 0].cpu().item()
                    )
            
            # FIXED: Update holdings with proper mask logic - preserve holdings in masked regions
            if attention_mask is not None:
                # Only update holdings for valid (non-masked) timesteps
                valid_mask = timestep_mask.expand_as(current_holding)
                new_holding = action * required_holding + (1 - action) * current_holding
                current_holding = valid_mask * new_holding + (1 - valid_mask) * current_holding
            else:
                # Standard update for fixed-length sequences
                current_holding = action * required_holding + (1 - action) * current_holding
        
        # Prepare results
        results = {
            'log_probs': torch.stack(log_probs, dim=1),
            'rewards': torch.stack(rewards, dim=1),
            'execution_probs': torch.stack(execution_probs, dim=1),
            'states': states,
            'current_holding': current_holding,
            'batch_size': batch_size,
            'n_timesteps': n_timesteps,
            'n_assets': n_assets
        }
        
        # Finalize tracking
        if self.enable_tracking:
            global_policy_tracker.finalize_episode()
            
        return results
    
    def get_name(self) -> str:
        return "policy_execution"

class AdvantageCalculationComponent(LossComponent):
    """Advantage calculation component using returns-to-go and value function baseline"""
    
    def compute_loss(self, log_probs: torch.Tensor, rewards: torch.Tensor, 
                    states: list, value_network=None, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute advantage-based policy loss"""
        
        # Handle rewards tensor which may include terminal cost (n_timesteps+1)
        batch_size, n_timesteps = log_probs.shape[:2]
        
        # Calculate returns-to-go for policy timesteps only
        returns_to_go = torch.zeros(batch_size, n_timesteps, 1, device=log_probs.device)
        
        for t in range(n_timesteps):
            # Sum all remaining rewards from timestep t to end (including terminal if present)
            returns_to_go[:, t, :] = torch.sum(rewards[:, t:], dim=1)
        
        # Compute value predictions and advantages if value function is available
        value_loss = torch.tensor(0.0, device=log_probs.device)
        if value_network is not None:
            value_predictions = []
            for t in range(n_timesteps):
                state = states[t]
                value_pred = value_network(
                    state['prev_holding'],
                    state['current_holding'], 
                    state['price'],
                    history=state.get('history_features'),
                    time_features=state.get('time_features'),
                    delta_features=state.get('delta_features')
                )
                value_predictions.append(value_pred)
            
            value_predictions = torch.stack(value_predictions, dim=1)
            
            # Calculate advantage: A(s,a) = returns_to_go - V(s)
            advantages = returns_to_go - value_predictions.detach()
            
            # Value function loss
            value_loss = torch.nn.functional.mse_loss(value_predictions, returns_to_go.detach())
        else:
            # Use returns-to-go directly as advantages (no baseline)
            advantages = returns_to_go
        
        # Policy gradient loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'advantages': advantages,
            'returns_to_go': returns_to_go
        }
    
    def get_name(self) -> str:
        return "advantage_calculation"

class TerminalCostComponent(LossComponent):
    """Terminal forced execution cost component"""
    
    def __init__(self, market_impact_scale=1.0, enable_tracking=False):
        self.market_impact_scale = market_impact_scale
        self.enable_tracking = enable_tracking
    
    def compute_loss(self, data: 'DataResult', current_holding: torch.Tensor, 
                    rewards: torch.Tensor, **kwargs) -> torch.Tensor:
        """Add terminal forced execution cost"""
        from utils.policy_tracker import global_policy_tracker
        from utils.logger import get_logger
        logger = get_logger(__name__)
        
        # Get terminal data with proper mask handling
        final_price, final_required_holding = self._get_terminal_data_with_mask(data, current_holding)
        final_holding_change = final_required_holding - current_holding
        
        # Terminal cost is always incurred (forced execution)
        final_cost = -(final_holding_change * final_price).sum(dim=-1, keepdim=True) * self.market_impact_scale
        
        # Add terminal cost to rewards
        terminal_reward = -final_cost
        enhanced_rewards = torch.cat([rewards, terminal_reward.unsqueeze(1)], dim=1)
        
        # Record terminal cost for tracking
        if self.enable_tracking and current_holding.shape[0] > 0:
            terminal_cost_value = final_cost[0, 0].cpu().item()
            global_policy_tracker.record_terminal_cost(terminal_cost_value)
        
        return enhanced_rewards
    
    def _get_terminal_data_with_mask(self, data, current_holding):
        """Get terminal data respecting attention mask for variable-length sequences"""
        prices = data.prices
        holdings = data.holdings
        attention_mask = data.metadata.get('attention_mask')
        
        if attention_mask is not None:
            batch_size, n_timesteps = attention_mask.shape
            final_prices = []
            final_holdings = []
            
            for b in range(batch_size):
                # Find last valid timestep for this batch item
                valid_mask = attention_mask[b, :]
                if valid_mask.sum() > 0:
                    last_valid_idx = valid_mask.nonzero(as_tuple=False)[-1].item()
                    final_prices.append(prices[b, last_valid_idx, :])
                    final_holdings.append(holdings[b, last_valid_idx, :])
                else:
                    # Fallback to last timestep if no valid timesteps found
                    final_prices.append(prices[b, -1, :])
                    final_holdings.append(holdings[b, -1, :])
            
            final_price = torch.stack(final_prices, dim=0)
            final_required_holding = torch.stack(final_holdings, dim=0)
        else:
            # Fixed-length sequences - use last timestep
            final_price = prices[:, -1, :]
            final_required_holding = holdings[:, -1, :]
        
        return final_price, final_required_holding
    
    def get_name(self) -> str:
        return "terminal_cost"

class LossCalculator:
    """
    Modular loss calculation system that combines multiple loss components
    with configurable weights for extensibility.
    """
    
    def __init__(self, config: Dict[str, float] = None):
        """
        Initialize loss calculator with component weights
        
        Args:
            config: Dictionary mapping loss component names to weights
                   e.g., {'policy_loss': 1.0, 'value_loss': 0.5, 'entropy_loss': 0.01}
        """
        self.components: Dict[str, LossComponent] = {}
        self.weights = config or {
            'policy_loss': 1.0,
            'value_loss': 0.5,
            'entropy_loss': 0.01
        }
    
    def add_component(self, component: LossComponent, weight: float = 1.0):
        """Add a loss component with specified weight"""
        name = component.get_name()
        self.components[name] = component
        if name not in self.weights:
            self.weights[name] = weight
    
    def remove_component(self, component_name: str):
        """Remove a loss component"""
        if component_name in self.components:
            del self.components[component_name]
        if component_name in self.weights:
            del self.weights[component_name]
    
    def set_weight(self, component_name: str, weight: float):
        """Update weight for a specific component"""
        self.weights[component_name] = weight
    
    def compute_total_loss(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute total weighted loss from all components
        
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        total_loss = torch.tensor(0.0)
        
        # Get device from first tensor argument
        device = None
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        
        if device is not None:
            total_loss = total_loss.to(device)
        
        # Compute each component loss
        for name, component in self.components.items():
            try:
                component_loss = component.compute_loss(**kwargs)
                losses[name] = component_loss
                
                # Add weighted component to total
                weight = self.weights.get(name, 0.0)
                total_loss += weight * component_loss
                
            except Exception as e:
                # Skip components that fail (e.g., missing required kwargs)
                print(f"Warning: Skipping {name} loss computation: {e}")
                continue
        
        losses['total_loss'] = total_loss
        return losses
    
    def compute_policy_loss_with_advantage(self, data: 'DataResult') -> Dict[str, torch.Tensor]:
        """
        Complete policy loss computation using modular components with proper mask handling
        
        This method orchestrates the entire policy execution simulation, terminal cost calculation,
        advantage computation, and final loss calculation using the modular component system.
        
        Args:
            data: DataResult containing prices, holdings, and metadata
            
        Returns:
            Dictionary containing all loss components and total loss
        """
        # Step 1: Execute complete policy simulation with proper mask logic
        policy_exec_component = self.components.get('policy_execution')
        if policy_exec_component is None:
            raise ValueError("PolicyExecutionComponent not found. Use create_advanced_loss_calculator() to initialize.")
        
        execution_results = policy_exec_component.compute_loss(data=data)
        
        # Step 2: Add terminal forced execution cost
        terminal_component = self.components.get('terminal_cost')
        if terminal_component is not None:
            enhanced_rewards = terminal_component.compute_loss(
                data=data,
                current_holding=execution_results['current_holding'],
                rewards=execution_results['rewards']
            )
        else:
            enhanced_rewards = execution_results['rewards']
        
        # Step 3: Calculate advantages and policy loss
        advantage_component = self.components.get('advantage_calculation')
        if advantage_component is not None:
            advantage_results = advantage_component.compute_loss(
                log_probs=execution_results['log_probs'],
                rewards=enhanced_rewards,
                states=execution_results['states'],
                value_network=policy_exec_component.value_network
            )
        else:
            # Fallback to simple REINFORCE without advantage
            total_reward = torch.sum(enhanced_rewards, dim=1, keepdim=True)
            policy_loss = -(execution_results['log_probs'] * total_reward.detach()).mean()
            advantage_results = {
                'policy_loss': policy_loss,
                'value_loss': torch.tensor(0.0, device=policy_loss.device)
            }
        
        # Step 4: Calculate entropy regularization
        entropy_component = self.components.get('entropy_loss')
        if entropy_component is not None:
            entropy_loss = entropy_component.compute_loss(
                execution_probs=execution_results['execution_probs']
            )
        else:
            entropy_loss = torch.tensor(0.0, device=execution_results['log_probs'].device)
        
        # Step 5: Calculate L2 regularization
        l2_component = self.components.get('l2_regularization')
        if l2_component is not None:
            l2_loss = l2_component.compute_loss()
        else:
            l2_loss = torch.tensor(0.0, device=execution_results['log_probs'].device)
        
        # Step 6: Combine all losses with weights
        policy_loss = advantage_results['policy_loss']
        value_loss = advantage_results['value_loss']
        
        # TEMPORARILY DISABLED VALUE LOSS (2025-08-26) for debugging cost-loss relationship
        # ANALYSIS RESULT: Value loss was dominating total loss (25+ out of 34.49 total)
        # With value loss disabled: Loss: 0.48, Cost: 0.04 - now in reasonable proportion
        # Original problem: Cost changes invisible in loss due to value loss dominance
        total_loss = (
            self.weights.get('policy_loss', 1.0) * policy_loss +
            # self.weights.get('value_loss', 0.5) * value_loss +  # DISABLED
            self.weights.get('entropy_loss', 0.01) * entropy_loss +
            self.weights.get('l2_regularization', 0.0001) * l2_loss
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'l2_regularization': l2_loss,
            'advantages': advantage_results.get('advantages'),
            'returns_to_go': advantage_results.get('returns_to_go')
        }
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of components and their weights"""
        return {name: self.weights.get(name, 0.0) for name in self.components.keys()}

def create_advanced_loss_calculator(policy_network: nn.Module, 
                                   value_network: Optional[nn.Module] = None,
                                   enable_tracking: bool = False,
                                   device: str = 'cpu',
                                   market_impact_scale: float = 1.0,
                                   base_execution_cost: float = 0.0,
                                   variable_execution_cost: float = 0.0,
                                   config: Optional[Dict[str, float]] = None) -> LossCalculator:
    """
    Create a complete loss calculator with all policy execution components
    
    Args:
        policy_network: Policy network for execution probability prediction
        value_network: Optional value function network for advantage calculation
        enable_tracking: Enable policy decision tracking for visualization
        device: Torch device for computations
        market_impact_scale: Scaling factor for market impact costs
        base_execution_cost: Fixed execution cost per action
        variable_execution_cost: Variable execution cost based on position size
        config: Optional weights configuration
        
    Returns:
        Configured LossCalculator instance with advanced components
    """
    default_config = {
        'policy_loss': 1.0,
        'value_loss': 0.5 if value_network is not None else 0.0,
        'entropy_loss': 0.01,
        'l2_regularization': 0.0001
    }
    
    if config:
        default_config.update(config)
    
    calculator = LossCalculator(default_config)
    
    # Add core components
    calculator.add_component(PolicyLossComponent())
    calculator.add_component(ValueLossComponent()) 
    calculator.add_component(EntropyLossComponent())
    calculator.add_component(L2RegularizationComponent(policy_network))
    
    # Add advanced execution components
    policy_exec_component = PolicyExecutionComponent(
        policy_network=policy_network,
        value_network=value_network,
        enable_tracking=enable_tracking,
        device=device,
        market_impact_scale=market_impact_scale,
        base_execution_cost=base_execution_cost,
        variable_execution_cost=variable_execution_cost
    )
    calculator.add_component(policy_exec_component)
    
    advantage_component = AdvantageCalculationComponent()
    calculator.add_component(advantage_component)
    
    terminal_component = TerminalCostComponent(
        market_impact_scale=market_impact_scale,
        enable_tracking=enable_tracking
    )
    calculator.add_component(terminal_component)
    
    return calculator

def create_default_loss_calculator(policy_network: nn.Module, 
                                 config: Optional[Dict[str, float]] = None) -> LossCalculator:
    """
    Create a loss calculator with default components (backward compatibility)
    
    Args:
        policy_network: Policy network for L2 regularization
        config: Optional weights configuration
        
    Returns:
        Configured LossCalculator instance
    """
    default_config = {
        'policy_loss': 1.0,
        'value_loss': 0.5,
        'entropy_loss': 0.01,
        'l2_regularization': 0.0001
    }
    
    if config:
        default_config.update(config)
    
    calculator = LossCalculator(default_config)
    
    # Add default components
    calculator.add_component(PolicyLossComponent())
    calculator.add_component(ValueLossComponent())
    calculator.add_component(EntropyLossComponent())
    calculator.add_component(L2RegularizationComponent(policy_network))
    
    return calculator