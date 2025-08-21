"""
Common interfaces and base classes for gamma hedge system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic
from dataclasses import dataclass
import torch

# Type variables for generic interfaces
T = TypeVar('T')
ConfigT = TypeVar('ConfigT')

@dataclass
class BaseConfig(ABC):
    """Base configuration abstract class"""
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration parameters"""
        pass
    
    @abstractmethod  
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary"""
        pass
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """Merge with another configuration"""
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot merge {type(other)} with {type(self)}")
        
        # Default implementation: override with other's values
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        merged = {**self_dict, **other_dict}
        return self.__class__.from_dict(merged)

class ConfigProvider(ABC):
    """Configuration provider interface"""
    
    @abstractmethod
    def get_config(self, config_type: str) -> BaseConfig:
        """Get configuration by type"""
        pass
    
    @abstractmethod
    def register_config(self, config_type: str, config: BaseConfig) -> None:
        """Register a configuration"""
        pass
    
    def has_config(self, config_type: str) -> bool:
        """Check if configuration exists"""
        try:
            self.get_config(config_type)
            return True
        except (KeyError, ValueError):
            return False

class DataResult:
    """Unified data result class - replaces MarketData"""
    
    def __init__(self, 
                 prices: torch.Tensor, 
                 holdings: torch.Tensor,
                 metadata: Optional[Dict[str, Any]] = None,
                 time_features: Optional[torch.Tensor] = None):
        # Validate tensor shapes match
        if prices.shape != holdings.shape:
            raise ValueError(f"Shape mismatch: prices {prices.shape} vs holdings {holdings.shape}")
        
        self.prices = prices
        self.holdings = holdings  
        self.metadata = metadata or {}
        self.time_features = time_features
        
    @property
    def shape(self) -> torch.Size:
        """Get tensor shape"""
        return self.prices.shape
        
    @property
    def n_assets(self) -> int:
        """Get number of assets"""
        return self.shape[-1]
        
    @property
    def sequence_length(self) -> Optional[int]:
        """Get sequence length if applicable"""
        return self.shape[-2] if len(self.shape) >= 2 else None
        
    @property
    def batch_size(self) -> Optional[int]:
        """Get batch size if applicable"""
        return self.shape[0] if len(self.shape) >= 3 else None
        
    def to(self, device: str) -> 'DataResult':
        """Move tensors to specified device"""
        return DataResult(
            self.prices.to(device),
            self.holdings.to(device),
            self.metadata.copy(),
            self.time_features.to(device) if self.time_features is not None else None
        )
        
    def extract_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract tensor pair for backward compatibility"""
        return self.prices, self.holdings
    
    def add_batch_dim(self) -> 'DataResult':
        """Add batch dimension"""
        return DataResult(
            self.prices.unsqueeze(0),
            self.holdings.unsqueeze(0),
            self.metadata.copy(),
            self.time_features.unsqueeze(0) if self.time_features is not None else None
        )
    
    def remove_batch_dim(self) -> 'DataResult':
        """Remove batch dimension"""
        return DataResult(
            self.prices.squeeze(0),
            self.holdings.squeeze(0),
            self.metadata.copy(),
            self.time_features.squeeze(0) if self.time_features is not None else None
        )

class DatasetInterface(ABC):
    """Unified dataset interface"""
    
    @abstractmethod
    def __len__(self) -> int:
        """Get dataset size"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> DataResult:
        """Get single sample"""
        pass
    
    @property
    @abstractmethod
    def n_assets(self) -> int:
        """Get number of assets"""
        pass
    
    @property  
    @abstractmethod
    def sequence_length(self) -> Optional[int]:
        """Get sequence length, None for variable length"""
        pass
    
    @abstractmethod
    def get_sample_data(self) -> DataResult:
        """Get sample data for shape validation"""
        pass
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        sample = self.get_sample_data()
        return {
            'dataset_size': len(self),
            'n_assets': self.n_assets,
            'sequence_length': self.sequence_length,
            'sample_shape': sample.shape,
            'data_type': type(self).__name__
        }

class DataProviderInterface(ABC):
    """Data provider interface"""
    
    @abstractmethod
    def create_dataset(self, config: BaseConfig, **kwargs) -> DatasetInterface:
        """Create dataset instance"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data provider is available"""
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        pass
    
    def validate_config(self, config: BaseConfig) -> bool:
        """Validate configuration for this provider"""
        return config.validate()