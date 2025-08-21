"""
Training layer specific configurations
"""

from dataclasses import dataclass
from typing import List
from common.interfaces import BaseConfig
from common.enhanced_config_manager import register_config_type

@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-3
    n_epochs: int = 10
    n_train_samples: int = 10000
    n_val_samples: int = 2000
    device: str = 'cuda'
    checkpoint_interval: int = 10
    early_stopping_patience: int = None
    
    # Regularization parameters - unified defaults based on typical usage
    entropy_weight: float = 0.01  # Default from trainer.py - encourages exploration
    base_execution_cost: float = 0.001  # Default from trainer.py - base transaction cost
    variable_execution_cost: float = 0.0001  # Default from trainer.py - variable transaction cost
    market_impact_scale: float = 0.01  # Scale factor for market impact cost to control reward magnitude
    
    # Value function parameters
    value_learning_rate: float = 1e-3  # Learning rate for value function
    value_loss_weight: float = 0.5  # Weight for value function loss in total loss
    value_hidden_dims: List[int] = None  # Hidden dimensions for value function network
    value_dropout_rate: float = 0.1  # Dropout rate for value function network
    
    def __post_init__(self):
        if self.value_hidden_dims is None:
            self.value_hidden_dims = [128, 64, 32]
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        return (self.batch_size > 0 and
                self.learning_rate > 0 and
                self.n_epochs > 0 and
                self.n_train_samples > 0 and
                self.n_val_samples > 0 and
                self.checkpoint_interval > 0 and
                self.device in ['cpu', 'cuda'] and
                self.entropy_weight >= 0 and
                self.base_execution_cost >= 0 and
                self.variable_execution_cost >= 0 and
                self.value_learning_rate > 0 and
                self.value_loss_weight >= 0 and
                self.value_dropout_rate >= 0 and
                self.value_dropout_rate <= 1 and
                all(dim > 0 for dim in self.value_hidden_dims))
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'n_train_samples': self.n_train_samples,
            'n_val_samples': self.n_val_samples,
            'device': self.device,
            'checkpoint_interval': self.checkpoint_interval,
            'early_stopping_patience': self.early_stopping_patience,
            'entropy_weight': self.entropy_weight,
            'base_execution_cost': self.base_execution_cost,
            'variable_execution_cost': self.variable_execution_cost,
            'value_learning_rate': self.value_learning_rate,
            'value_loss_weight': self.value_loss_weight,
            'value_hidden_dims': self.value_hidden_dims,
            'value_dropout_rate': self.value_dropout_rate
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create from dictionary"""
        return cls(**config_dict)

@dataclass
class ModelConfig(BaseConfig):
    """Model configuration"""
    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    input_dim: int = None  # Will be calculated dynamically
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        return (self.dropout_rate >= 0 and 
                self.dropout_rate <= 1 and
                all(dim > 0 for dim in self.hidden_dims))
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'input_dim': self.input_dim
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**config_dict)

@dataclass
class DeltaHedgeConfig(BaseConfig):
    """Delta hedge configuration"""
    enable_delta: bool = False
    underlying_codes: List[str] = None
    weekly_codes: List[str] = None
    sequence_length: int = 100
    
    def __post_init__(self):
        if self.underlying_codes is None:
            self.underlying_codes = ['USU5', 'FVU5']
        if self.weekly_codes is None:
            self.weekly_codes = ['3CN5']
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        return (self.sequence_length > 0 and
                len(self.underlying_codes) > 0 and
                len(self.weekly_codes) > 0)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'enable_delta': self.enable_delta,
            'underlying_codes': self.underlying_codes,
            'weekly_codes': self.weekly_codes,
            'sequence_length': self.sequence_length
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'DeltaHedgeConfig':
        """Create from dictionary"""
        return cls(**config_dict)

# Register configuration types globally
register_config_type('training', TrainingConfig)
register_config_type('model', ModelConfig)
register_config_type('delta_hedge', DeltaHedgeConfig)