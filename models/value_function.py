import torch
import torch.nn as nn

class ValueFunction:
    """
    Implements the value function V(t; W, A) using dynamic programming
    """
    
    def __init__(self, policy_network: nn.Module):
        self.policy = policy_network
    
    def compute_cost(self, 
                     holding_change: torch.Tensor,
                     price: torch.Tensor) -> torch.Tensor:
        """
        Compute execution cost c(t) = -(W(t) - W(t-)) · A(t)
        """
        return -(holding_change * price).sum(dim=-1)
    
    def evaluate_policy(self,
                       prices: torch.Tensor,
                       holdings: torch.Tensor,
                       initial_holding: torch.Tensor) -> torch.Tensor:
        """
        Evaluate policy on a trajectory
        Args:
            prices: Asset prices (batch_size, n_timesteps, n_assets)
            holdings: Required holdings (batch_size, n_timesteps, n_assets)
            initial_holding: Initial holdings (batch_size, n_assets)
        Returns:
            Total cost (batch_size,)
        """
        batch_size, n_timesteps, n_assets = prices.shape
        device = prices.device
        
        total_cost = torch.zeros(batch_size, device=device)
        current_holding = initial_holding.clone()
        
        for t in range(n_timesteps):
            # Get current price and required holding
            current_price = prices[:, t, :]
            required_holding = holdings[:, t, :]
            
            # Compute policy decision
            with torch.no_grad():
                exec_prob = self.policy(
                    current_holding,
                    required_holding,
                    current_price
                )
            
            # Sample execution decision
            execute = torch.bernoulli(exec_prob).squeeze(-1)
            
            # Compute cost for those who execute
            holding_change = required_holding - current_holding
            cost = self.compute_cost(holding_change, current_price)
            total_cost += execute * cost
            
            # Update holdings for those who executed
            execute_mask = execute.unsqueeze(-1).expand_as(current_holding)
            current_holding = torch.where(
                execute_mask > 0,
                required_holding,
                current_holding
            )
        
        # Final execution at T (mandatory)
        final_price = prices[:, -1, :]
        final_holding = holdings[:, -1, :]
        final_cost = self.compute_cost(
            final_holding - current_holding,
            final_price
        )
        total_cost += final_cost
        
        return total_cost