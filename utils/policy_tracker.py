import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class HedgeDecision:
    """
    Data structure to store a single hedge decision
    """
    timestep: int
    execution_probability: float
    action_taken: bool  # True if hedge executed, False if waiting
    current_price: float
    option_price: float
    future_price: float
    current_holding: float
    required_holding: float
    holding_change: float
    immediate_cost: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

class PolicyTracker:
    """
    Tracks policy decisions during training and evaluation for visualization
    """
    
    def __init__(self, enable_tracking: bool = True):
        self.enable_tracking = enable_tracking
        self.current_episode_decisions: List[HedgeDecision] = []
        self.all_episodes: List[List[HedgeDecision]] = []
        self.episode_metadata: List[Dict] = []
    
    def start_new_episode(self, metadata: Optional[Dict] = None):
        """Start tracking a new episode"""
        if not self.enable_tracking:
            return
            
        # Save previous episode if it has decisions
        if self.current_episode_decisions:
            self.all_episodes.append(self.current_episode_decisions.copy())
            self.episode_metadata.append(metadata or {})
        
        # Reset for new episode
        self.current_episode_decisions = []
    
    def record_decision(self,
                       timestep: int,
                       execution_probability: float,
                       action_taken: bool,
                       current_price: torch.Tensor,
                       current_holding: torch.Tensor,
                       required_holding: torch.Tensor,
                       immediate_cost: float,
                       option_price: Optional[float] = None,
                       future_price: Optional[float] = None,
                       greeks: Optional[Dict[str, float]] = None):
        """
        Record a single hedge decision
        
        Args:
            timestep: Current time step
            execution_probability: Policy output probability
            action_taken: Whether hedge was executed
            current_price: Asset price tensor
            current_holding: Current position tensor
            required_holding: Required position tensor
            immediate_cost: Cost of the action
            option_price: Option price if available
            future_price: Future price if available
            greeks: Option Greeks if available
        """
        if not self.enable_tracking:
            return
        
        # Convert tensors to scalars (assuming single asset for now)
        if isinstance(current_price, torch.Tensor):
            price_val = current_price.cpu().item() if current_price.numel() == 1 else current_price.cpu().numpy()[0]
        else:
            price_val = float(current_price)
        
        if isinstance(current_holding, torch.Tensor):
            curr_hold = current_holding.cpu().item() if current_holding.numel() == 1 else current_holding.cpu().numpy()[0]
        else:
            curr_hold = float(current_holding)
            
        if isinstance(required_holding, torch.Tensor):
            req_hold = required_holding.cpu().item() if required_holding.numel() == 1 else required_holding.cpu().numpy()[0]
        else:
            req_hold = float(required_holding)
        
        holding_change = req_hold - curr_hold
        
        # Extract Greeks if provided
        delta = greeks.get('delta', None) if greeks else None
        gamma = greeks.get('gamma', None) if greeks else None
        theta = greeks.get('theta', None) if greeks else None
        vega = greeks.get('vega', None) if greeks else None
        
        decision = HedgeDecision(
            timestep=timestep,
            execution_probability=execution_probability,
            action_taken=action_taken,
            current_price=price_val,
            option_price=option_price or price_val,  # Fallback to current price
            future_price=future_price or price_val,  # Fallback to current price
            current_holding=curr_hold,
            required_holding=req_hold,
            holding_change=holding_change,
            immediate_cost=immediate_cost,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega
        )
        
        self.current_episode_decisions.append(decision)
    
    def record_terminal_cost(self, terminal_cost: float):
        """Record terminal forced execution cost for the current episode"""
        if not self.enable_tracking or not self.current_episode_decisions:
            return
        
        # Store terminal cost in episode metadata
        if not hasattr(self, '_current_terminal_cost'):
            self._current_terminal_cost = 0.0
        self._current_terminal_cost += terminal_cost
    
    def finalize_episode(self, metadata: Optional[Dict] = None):
        """Finalize current episode and prepare for analysis"""
        if not self.enable_tracking or not self.current_episode_decisions:
            return
        
        # Include terminal cost in episode metadata
        episode_meta = metadata or {}
        if hasattr(self, '_current_terminal_cost'):
            episode_meta['terminal_cost'] = self._current_terminal_cost
            self._current_terminal_cost = 0.0  # Reset for next episode
        
        self.all_episodes.append(self.current_episode_decisions.copy())
        self.episode_metadata.append(episode_meta)
        self.current_episode_decisions = []
    
    def get_latest_episode(self) -> List[HedgeDecision]:
        """Get decisions from the most recent episode"""
        if self.current_episode_decisions:
            return self.current_episode_decisions
        elif self.all_episodes:
            return self.all_episodes[-1]
        else:
            return []
    
    def get_episode(self, episode_idx: int) -> List[HedgeDecision]:
        """Get decisions from a specific episode"""
        if episode_idx < len(self.all_episodes):
            return self.all_episodes[episode_idx]
        else:
            return []
    
    def get_hedge_summary(self, episode_idx: Optional[int] = None) -> Dict:
        """
        Get summary statistics for hedge decisions
        
        Args:
            episode_idx: Specific episode index, or None for latest episode
        
        Returns:
            Dictionary with summary statistics
        """
        if episode_idx is not None:
            decisions = self.get_episode(episode_idx)
        else:
            decisions = self.get_latest_episode()
        
        if not decisions:
            return {}
        
        hedge_points = [d for d in decisions if d.action_taken]
        wait_points = [d for d in decisions if not d.action_taken]
        
        total_cost = sum(d.immediate_cost for d in decisions if d.action_taken)
        
        # Add terminal cost if available
        if episode_idx is not None and episode_idx < len(self.episode_metadata):
            terminal_cost = self.episode_metadata[episode_idx].get('terminal_cost', 0)
        elif len(self.episode_metadata) > 0:
            terminal_cost = self.episode_metadata[-1].get('terminal_cost', 0)
        else:
            terminal_cost = 0
        
        total_cost += terminal_cost
        
        summary = {
            'total_decisions': len(decisions),
            'hedge_actions': len(hedge_points),
            'wait_actions': len(wait_points),
            'hedge_ratio': len(hedge_points) / len(decisions) if decisions else 0,
            'total_cost': total_cost,
            'avg_execution_prob': np.mean([d.execution_probability for d in decisions]),
            'hedge_timesteps': [d.timestep for d in hedge_points],
            'hedge_prices': [d.current_price for d in hedge_points],
            'hedge_costs': [d.immediate_cost for d in hedge_points]
        }
        
        if hedge_points:
            summary.update({
                'avg_hedge_cost': np.mean([d.immediate_cost for d in hedge_points]),
                'avg_hedge_price': np.mean([d.current_price for d in hedge_points]),
                'hedge_price_range': (
                    min(d.current_price for d in hedge_points),
                    max(d.current_price for d in hedge_points)
                )
            })
        
        return summary
    
    def clear(self):
        """Clear all recorded episodes"""
        self.current_episode_decisions = []
        self.all_episodes = []
        self.episode_metadata = []
    
    def export_to_dataframe(self, episode_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Export episode decisions to a pandas DataFrame for analysis
        
        Args:
            episode_idx: Specific episode index, or None for all episodes
        
        Returns:
            DataFrame with all decision data
        """
        if episode_idx is not None:
            episodes_to_export = [self.get_episode(episode_idx)]
            episode_indices = [episode_idx]
        else:
            episodes_to_export = self.all_episodes + ([self.current_episode_decisions] if self.current_episode_decisions else [])
            episode_indices = list(range(len(episodes_to_export)))
        
        all_data = []
        for ep_idx, episode in enumerate(episodes_to_export):
            for decision in episode:
                row = {
                    'episode': episode_indices[ep_idx],
                    'timestep': decision.timestep,
                    'execution_probability': decision.execution_probability,
                    'action_taken': decision.action_taken,
                    'current_price': decision.current_price,
                    'option_price': decision.option_price,
                    'future_price': decision.future_price,
                    'current_holding': decision.current_holding,
                    'required_holding': decision.required_holding,
                    'holding_change': decision.holding_change,
                    'immediate_cost': decision.immediate_cost,
                    'delta': decision.delta,
                    'gamma': decision.gamma,
                    'theta': decision.theta,
                    'vega': decision.vega
                }
                all_data.append(row)
        
        return pd.DataFrame(all_data)

# Global instance for easy access
global_policy_tracker = PolicyTracker()