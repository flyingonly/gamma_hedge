import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PolicyNetwork(nn.Module):
    """
    Neural network for execution policy u*(θ; W(t_{i-1}), W, S, H, Δ)
    
    Enhanced with delta features for improved time sensitivity:
    - Basic features: prev_holding, current_holding, price (3 dims)
    - Time features: time_remaining (1 dim)
    - History features: market history (14 dims)
    - Delta features: current_delta, delta_change, delta_acceleration (3 dims)
    
    Total input dimensions: 3 + 1 + 14 + 3 = 21 dims (when all features enabled)
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.1):
        super(PolicyNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Build hidden layers (remove BatchNorm to handle small batch sizes)
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (execution probability)
        self.output_layer = nn.Linear(prev_dim, 1)
        
    def forward(self, 
                prev_holding: torch.Tensor,
                current_holding: torch.Tensor,
                price: torch.Tensor,
                history: Optional[torch.Tensor] = None,
                time_features: Optional[torch.Tensor] = None,
                delta_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        Args:
            prev_holding: Previous holdings W(t_{i-1})
            current_holding: Current required holdings W(t_i)
            price: Current asset prices A(t_i)
            history: Market history H (optional)
            time_features: Time-based features like time_remaining (optional)
            delta_features: Delta-based features [current_delta, delta_change, delta_acceleration] (optional)
        Returns:
            Execution probability (0 to 1)
        """
        # Concatenate inputs
        inputs = [prev_holding, current_holding, price]
        if history is not None:
            inputs.append(history)
        if time_features is not None:
            inputs.append(time_features)
        if delta_features is not None:
            inputs.append(delta_features)
        
        x = torch.cat(inputs, dim=-1)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        # Get execution probability
        logits = self.output_layer(x)
        prob = torch.sigmoid(logits)
        
        return prob