"""
Standard Data Exchange Interfaces for Gamma Hedge System

This module defines the core interfaces and data structures used for 
communication between system modules, ensuring loose coupling and 
consistent data formats.

Key Principles:
- Simple, well-defined data structures
- Protocol-based interfaces for flexibility
- Immutable data objects where possible
- Clear separation of concerns
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Protocol, Tuple, List
from abc import ABC, abstractmethod
import torch
from enum import Enum


class DataMode(Enum):
    """Data processing modes"""
    SPARSE = "sparse"
    DENSE_INTERPOLATED = "dense_interpolated"
    DENSE_DAILY_RECALC = "dense_daily_recalc"


class TrainingPhase(Enum):
    """Training phase indicators"""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class TrainingData:
    """
    Standard data format for training and evaluation
    
    This replaces multiple inconsistent data formats used across
    the system with a single, well-defined structure.
    """
    prices: torch.Tensor          # Shape: (batch_size, sequence_length, n_assets)
    positions: torch.Tensor       # Shape: (batch_size, sequence_length, n_assets)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validation
        if self.prices.dim() != 3:
            raise ValueError(f"prices must be 3D tensor, got shape {self.prices.shape}")
        if self.positions.dim() != 3:
            raise ValueError(f"positions must be 3D tensor, got shape {self.positions.shape}")
        if self.prices.shape != self.positions.shape:
            raise ValueError(f"prices and positions must have same shape, got {self.prices.shape} vs {self.positions.shape}")
    
    @property
    def batch_size(self) -> int:
        return self.prices.shape[0]
    
    @property
    def sequence_length(self) -> int:
        return self.prices.shape[1]
    
    @property
    def n_assets(self) -> int:
        return self.prices.shape[2]
    
    def to(self, device: torch.device) -> 'TrainingData':
        """Move data to specified device"""
        return TrainingData(
            prices=self.prices.to(device),
            positions=self.positions.to(device),
            metadata=self.metadata
        )
    
    def detach(self) -> 'TrainingData':
        """Detach tensors from computation graph"""
        return TrainingData(
            prices=self.prices.detach(),
            positions=self.positions.detach(),
            metadata=self.metadata
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Standard format for evaluation results"""
    total_cost: float
    cost_components: Dict[str, float] = field(default_factory=dict)
    execution_stats: Dict[str, Any] = field(default_factory=dict)
    policy_decisions: Optional[torch.Tensor] = None
    
    @property
    def cost_std(self) -> float:
        """Standard deviation of cost (for compatibility)"""
        return self.cost_components.get('std', 0.0)


@dataclass(frozen=True)
class PortfolioPosition:
    """Standard format for portfolio positions"""
    option_key: str        # Format: "3CN5/CALL_111.0"
    position_size: float   # Positive for long, negative for short
    
    @property
    def weekly_code(self) -> str:
        return self.option_key.split('/')[0]
    
    @property
    def option_type(self) -> str:
        return self.option_key.split('/')[1].split('_')[0]
    
    @property
    def strike(self) -> float:
        return float(self.option_key.split('/')[1].split('_')[1])


class DataProvider(Protocol):
    """
    Protocol for data providers
    
    All data loading and processing components should implement this
    interface to ensure consistent behavior across the system.
    """
    
    def get_training_data(self, 
                         batch_size: int,
                         sequence_length: int,
                         phase: TrainingPhase = TrainingPhase.TRAIN) -> TrainingData:
        """
        Get training data batch
        
        Args:
            batch_size: Number of sequences in batch
            sequence_length: Length of each sequence
            phase: Training phase (train/validation/test)
            
        Returns:
            TrainingData object with specified dimensions
        """
        ...
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about available data"""
        ...


class ConfigProvider(Protocol):
    """Protocol for configuration providers"""
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value"""
        ...
    
    def validate(self) -> None:
        """Validate configuration"""
        ...


class TrainingEngine(Protocol):
    """Protocol for training engines"""
    
    def train(self, 
             data_provider: DataProvider,
             n_epochs: int,
             validation_data: Optional[DataProvider] = None) -> Dict[str, Any]:
        """
        Train model with provided data
        
        Args:
            data_provider: Training data provider
            n_epochs: Number of training epochs
            validation_data: Optional validation data provider
            
        Returns:
            Training history and metrics
        """
        ...
    
    def evaluate(self, data_provider: DataProvider) -> EvaluationResult:
        """Evaluate model performance"""
        ...


class Visualizer(Protocol):
    """Protocol for visualization components"""
    
    def create_visualization(self, data: Any, title: str = "") -> Any:
        """Create visualization from data"""
        ...
    
    def save_visualization(self, output_path: str) -> None:
        """Save visualization to file"""
        ...


# Abstract base classes for common implementations

class BaseDataProcessor(ABC):
    """Base class for data processors"""
    
    @abstractmethod
    def process_data(self, raw_data: Any) -> TrainingData:
        """Process raw data into standard format"""
        pass
    
    @abstractmethod
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed data"""
        pass


# Backward compatibility classes

@dataclass
class DataResult:
    """
    Legacy compatibility class for DataResult
    
    This provides backward compatibility with existing code that uses
    the DataResult format. New code should use TrainingData instead.
    
    Enhanced with delta_features for time-sensitive training.
    """
    prices: torch.Tensor
    holdings: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)
    time_features: Optional[torch.Tensor] = None
    history_features: Optional[torch.Tensor] = None
    delta_features: Optional[torch.Tensor] = None  # Shape: (batch, seq, n_delta_features)
    
    def __post_init__(self):
        # Ensure 3D tensors
        if self.prices.dim() == 2:
            self.prices = self.prices.unsqueeze(0)  # Add batch dimension
        if self.holdings.dim() == 2:
            self.holdings = self.holdings.unsqueeze(0)  # Add batch dimension
            
        # Validation
        if self.prices.shape != self.holdings.shape:
            raise ValueError(f"prices and holdings must have same shape, got {self.prices.shape} vs {self.holdings.shape}")
    
    @property
    def n_samples(self) -> int:
        return self.prices.shape[0]
    
    @property
    def sequence_length(self) -> int:
        return self.prices.shape[1] if self.prices.dim() > 1 else 1
    
    @property
    def n_assets(self) -> int:
        return self.prices.shape[2] if self.prices.dim() > 2 else self.prices.shape[1] if self.prices.dim() > 1 else 1
    
    def to(self, device) -> 'DataResult':
        """Move to device"""
        return DataResult(
            prices=self.prices.to(device),
            holdings=self.holdings.to(device),
            metadata=self.metadata,
            time_features=self.time_features.to(device) if self.time_features is not None else None,
            history_features=self.history_features.to(device) if self.history_features is not None else None,
            delta_features=self.delta_features.to(device) if self.delta_features is not None else None
        )
    
    def __getitem__(self, idx) -> 'DataResult':
        """Support indexing"""
        return DataResult(
            prices=self.prices[idx:idx+1] if isinstance(idx, int) else self.prices[idx],
            holdings=self.holdings[idx:idx+1] if isinstance(idx, int) else self.holdings[idx],
            metadata=self.metadata,
            time_features=self.time_features[idx:idx+1] if isinstance(idx, int) and self.time_features is not None else self.time_features[idx] if self.time_features is not None else None,
            history_features=self.history_features[idx:idx+1] if isinstance(idx, int) and self.history_features is not None else self.history_features[idx] if self.history_features is not None else None,
            delta_features=self.delta_features[idx:idx+1] if isinstance(idx, int) and self.delta_features is not None else self.delta_features[idx] if self.delta_features is not None else None
        )
    
    def extract_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract tensors for compatibility"""
        return self.prices, self.holdings


class DatasetInterface(ABC):
    """Legacy compatibility interface"""
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod  
    def __getitem__(self, idx) -> DataResult:
        pass


class DataProviderInterface(ABC):
    """Legacy compatibility interface"""
    
    @abstractmethod
    def get_data(self, *args, **kwargs) -> DataResult:
        pass


# Utility functions for interface conversion

def convert_legacy_data(prices: torch.Tensor, 
                       holdings: torch.Tensor,
                       metadata: Optional[Dict[str, Any]] = None) -> TrainingData:
    """
    Convert legacy data format to standard TrainingData
    
    Args:
        prices: Legacy price tensor  
        holdings: Legacy holdings tensor
        metadata: Optional metadata dictionary
        
    Returns:
        TrainingData object in standard format
    """
    return TrainingData(
        prices=prices,
        positions=holdings,
        metadata=metadata or {}
    )


def extract_legacy_format(data: TrainingData) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract data in legacy format for backward compatibility
    
    Args:
        data: TrainingData object
        
    Returns:
        Tuple of (prices, holdings) tensors
    """
    return data.prices, data.positions


def create_portfolio_from_dict(positions: Dict[str, float]) -> List[PortfolioPosition]:
    """
    Create portfolio positions from dictionary format
    
    Args:
        positions: Dictionary mapping option keys to position sizes
        
    Returns:
        List of PortfolioPosition objects
    """
    return [
        PortfolioPosition(option_key=key, position_size=size)
        for key, size in positions.items()
    ]


def portfolio_to_dict(portfolio: List[PortfolioPosition]) -> Dict[str, float]:
    """
    Convert portfolio positions to dictionary format
    
    Args:
        portfolio: List of PortfolioPosition objects
        
    Returns:
        Dictionary mapping option keys to position sizes
    """
    return {pos.option_key: pos.position_size for pos in portfolio}