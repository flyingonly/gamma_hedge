import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.value_function import ValueFunction
from data.data_types import MarketData

class Evaluator:
    """Evaluate trained policy performance"""
    
    def __init__(self, policy_network: nn.Module, device: str = 'cuda'):
        self.policy = policy_network.to(device)
        self.device = device
        self.value_func = ValueFunction(policy_network)
    
    def evaluate_trajectory(self,
                           data: Union[MarketData, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Evaluate policy on a single trajectory"""
        self.policy.eval()
        
        with torch.no_grad():
            # Handle both MarketData and tuple formats
            if isinstance(data, MarketData):
                prices = data.prices.to(self.device)
                holdings = data.holdings.to(self.device)
            else:
                # Legacy tuple format
                prices, holdings = data
                prices = prices.to(self.device)
                holdings = holdings.to(self.device)
            
            # Get initial holding
            if len(holdings.shape) == 3:
                initial_holding = holdings[:, 0, :]
            else:
                initial_holding = holdings[0].unsqueeze(0)
                prices = prices.unsqueeze(0)
                holdings = holdings.unsqueeze(0)
            
            # Evaluate policy
            total_cost = self.value_func.evaluate_policy(
                prices, holdings, initial_holding
            )
            
            # Compute statistics
            metrics = {
                'total_cost': total_cost.mean().item(),
                'cost_std': total_cost.std().item() if total_cost.numel() > 1 else 0,
                'min_cost': total_cost.min().item(),
                'max_cost': total_cost.max().item()
            }
        
        return metrics
    
    def compare_policies(self,
                        data: Union[MarketData, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
        """Compare trained policy vs baseline strategies"""
        results = {}
        
        # Handle data format conversion for legacy baseline methods
        if isinstance(data, MarketData):
            prices = data.prices
            holdings = data.holdings
        else:
            prices, holdings = data
        
        # Evaluate trained policy
        results['trained'] = self.evaluate_trajectory(data)
        
        # Evaluate "execute all" baseline
        results['execute_all'] = self._evaluate_execute_all(prices, holdings)
        
        # Evaluate "execute at end" baseline
        results['execute_end'] = self._evaluate_execute_end(prices, holdings)
        
        return results
    
    def _evaluate_execute_all(self,
                             prices: torch.Tensor,
                             holdings: torch.Tensor) -> Dict[str, float]:
        """Baseline: Execute at every timestep"""
        batch_size = prices.shape[0] if len(prices.shape) == 3 else 1
        prices = prices.to(self.device)
        holdings = holdings.to(self.device)
        
        if len(prices.shape) == 2:
            prices = prices.unsqueeze(0)
            holdings = holdings.unsqueeze(0)
        
        total_cost = torch.zeros(batch_size, device=self.device)
        current_holding = holdings[:, 0, :]
        
        for t in range(prices.shape[1]):
            required_holding = holdings[:, t, :]
            holding_change = required_holding - current_holding
            cost = -(holding_change * prices[:, t, :]).sum(dim=-1)
            total_cost += cost
            current_holding = required_holding
        
        return {
            'total_cost': total_cost.mean().item(),
            'cost_std': total_cost.std().item() if batch_size > 1 else 0
        }
    
    def _evaluate_execute_end(self,
                             prices: torch.Tensor,
                             holdings: torch.Tensor) -> Dict[str, float]:
        """Baseline: Execute only at the end"""
        if len(prices.shape) == 2:
            prices = prices.unsqueeze(0)
            holdings = holdings.unsqueeze(0)
        
        initial_holding = holdings[:, 0, :]
        final_holding = holdings[:, -1, :]
        final_price = prices[:, -1, :]
        
        cost = -(final_holding - initial_holding) * final_price
        total_cost = cost.sum(dim=-1)
        
        return {
            'total_cost': total_cost.mean().item(),
            'cost_std': total_cost.std().item() if total_cost.numel() > 1 else 0
        }