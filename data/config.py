"""
Data layer specific configurations
Moved from utils/config.py to break circular dependency
"""

from dataclasses import dataclass
from typing import List
from common.interfaces import BaseConfig
from common.enhanced_config_manager import register_config_type

@dataclass
class MarketConfig(BaseConfig):
    """Market simulation configuration"""
    n_assets: int = 3
    n_timesteps: int = 50
    initial_price: float = 100.0
    volatility: float = 0.2
    dt: float = 0.01
    drift: float = 0.05
    initial_holding: List[float] = None
    target_holding: List[float] = None
    
    def __post_init__(self):
        if self.initial_holding is None:
            self.initial_holding = [0.0] * self.n_assets
        if self.target_holding is None:
            self.target_holding = [1.0] * self.n_assets
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        return (self.n_assets > 0 and 
                self.n_timesteps > 0 and 
                self.volatility >= 0 and 
                self.dt > 0 and
                len(self.initial_holding) == self.n_assets and
                len(self.target_holding) == self.n_assets)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'n_assets': self.n_assets,
            'n_timesteps': self.n_timesteps,
            'initial_price': self.initial_price,
            'volatility': self.volatility,
            'dt': self.dt,
            'drift': self.drift,
            'initial_holding': self.initial_holding,
            'target_holding': self.target_holding
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MarketConfig':
        """Create from dictionary"""
        return cls(**config_dict)

# Register this configuration type globally
register_config_type('market', MarketConfig)