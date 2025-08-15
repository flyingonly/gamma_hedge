import numpy as np
import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class MarketConfig:
    """Configuration for market simulation"""
    n_assets: int = 1
    n_timesteps: int = 10
    dt: float = 0.01
    initial_price: float = 100.0
    volatility: float = 0.2
    drift: float = 0.05
    initial_holding: float = 5.0
    target_holding: float = 7.0
    seed: Optional[int] = None

class MarketSimulator:
    """Simulates market price dynamics and holding requirements"""
    
    def __init__(self, config: MarketConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
    
    def generate_price_paths(self, n_paths: int) -> torch.Tensor:
        """
        Generate price paths using Geometric Brownian Motion
        Returns: Tensor of shape (n_paths, n_timesteps, n_assets)
        """
        dt = self.config.dt
        n_steps = self.config.n_timesteps
        n_assets = self.config.n_assets
        
        # Generate random shocks
        z = np.random.randn(n_paths, n_steps, n_assets)
        
        # Initialize price paths
        prices = np.zeros((n_paths, n_steps + 1, n_assets))
        prices[:, 0, :] = self.config.initial_price
        
        # Generate price paths using GBM
        for t in range(1, n_steps + 1):
            dW = np.sqrt(dt) * z[:, t-1, :]
            prices[:, t, :] = prices[:, t-1, :] * (
                1 + self.config.drift * dt + self.config.volatility * dW
            )
        
        return torch.tensor(prices, dtype=torch.float32)
    
    def generate_holding_schedule(self, n_paths: int) -> torch.Tensor:
        """
        Generate holding schedule W(t) for each path
        Returns: Tensor of shape (n_paths, n_timesteps, n_assets)
        """
        n_steps = self.config.n_timesteps
        n_assets = self.config.n_assets
        
        holdings = np.zeros((n_paths, n_steps + 1, n_assets))
        
        # Linear interpolation from initial to target holding
        for i in range(n_steps + 1):
            alpha = i / n_steps
            holdings[:, i, :] = (1 - alpha) * self.config.initial_holding + \
                               alpha * self.config.target_holding
        
        # Add some noise to make it more realistic
        noise = np.random.randn(n_paths, n_steps + 1, n_assets) * 0.5
        holdings = holdings + noise
        
        # Ensure final holding matches target
        holdings[:, -1, :] = self.config.target_holding
        
        return torch.tensor(holdings, dtype=torch.float32)
    
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of price paths and holding schedules"""
        prices = self.generate_price_paths(batch_size)
        holdings = self.generate_holding_schedule(batch_size)
        return prices, holdings