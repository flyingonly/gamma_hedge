import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional
from common.interfaces import DataResult

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
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of components and their weights"""
        return {name: self.weights.get(name, 0.0) for name in self.components.keys()}

def create_default_loss_calculator(policy_network: nn.Module, 
                                 config: Optional[Dict[str, float]] = None) -> LossCalculator:
    """
    Create a loss calculator with default components
    
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