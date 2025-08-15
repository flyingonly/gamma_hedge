from dataclasses import dataclass
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-3
    n_epochs: int = 10
    n_train_samples: int = 10000
    n_val_samples: int = 2000
    device: str = 'cuda'

@dataclass
class ModelConfig:
    """Model configuration"""
    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]

def get_config():
    """Get default configuration"""
    from data.market_simulator import MarketConfig
    
    return {
        'market': MarketConfig(),
        'training': TrainingConfig(),
        'model': ModelConfig()
    }