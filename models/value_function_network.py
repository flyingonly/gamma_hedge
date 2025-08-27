import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ValueFunctionNetwork(nn.Module):
    """
    Neural network for value function V(s) that predicts expected returns
    from current state. Takes the same inputs as PolicyNetwork but outputs
    a single value representing expected cumulative reward.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.1):
        super(ValueFunctionNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Build hidden layers (consistent with PolicyNetwork architecture)
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (single value prediction)
        self.output_layer = nn.Linear(prev_dim, 1)
        
    def forward(self, 
                prev_holding: torch.Tensor,
                current_holding: torch.Tensor,
                price: torch.Tensor,
                history: Optional[torch.Tensor] = None,
                time_features: Optional[torch.Tensor] = None,
                delta_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to predict value function V(s)
        
        Args:
            prev_holding: Previous holdings W(t_{i-1})
            current_holding: Current required holdings W(t_i) 
            price: Current asset prices A(t_i)
            history: Market history H (optional)
            time_features: Time-based features like time_remaining (optional)
            delta_features: Delta-based features [current_delta, delta_change, delta_acceleration] (optional)
            
        Returns:
            Expected cumulative reward from current state (single value)
        """
        # Concatenate inputs (same as PolicyNetwork)
        inputs = [prev_holding, current_holding, price]
        if history is not None:
            inputs.append(history)
        if time_features is not None:
            inputs.append(time_features)
        if delta_features is not None:
            inputs.append(delta_features)
            
        x = torch.cat(inputs, dim=-1)
        
        # Forward through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer - no activation function for value prediction
        value = self.output_layer(x)
        
        return value
    
    def compute_value_loss(self, 
                          predicted_values: torch.Tensor,
                          target_returns: torch.Tensor) -> torch.Tensor:
        """
        Compute value function loss (MSE between predicted and actual returns)
        
        Args:
            predicted_values: Network predictions V(s)
            target_returns: Actual returns-to-go
            
        Returns:
            MSE loss
        """
        return F.mse_loss(predicted_values, target_returns.detach())