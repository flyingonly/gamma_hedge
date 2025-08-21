"""
Data types and base classes for unified data loading interface
"""

import torch
from dataclasses import dataclass
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


@dataclass
class MarketData:
    """Unified data container for market information"""
    prices: torch.Tensor      # (sequence_length, n_assets) or (batch_size, sequence_length, n_assets)
    holdings: torch.Tensor    # (sequence_length, n_assets) or (batch_size, sequence_length, n_assets)
    metadata: Optional[Dict[str, Any]] = None  # For future extensions
    
    def __post_init__(self):
        """Validate data consistency"""
        if self.prices.shape != self.holdings.shape:
            raise ValueError(f"Prices shape {self.prices.shape} doesn't match holdings shape {self.holdings.shape}")
    
    def to(self, device: str) -> 'MarketData':
        """Move data to specified device"""
        return MarketData(
            prices=self.prices.to(device),
            holdings=self.holdings.to(device),
            metadata=self.metadata
        )
    
    @property
    def shape(self) -> torch.Size:
        """Get data shape"""
        return self.prices.shape
    
    @property
    def n_assets(self) -> int:
        """Get number of assets"""
        return self.prices.shape[-1]
    
    @property
    def sequence_length(self) -> int:
        """Get sequence length"""
        return self.prices.shape[-2]
    
    def add_batch_dim(self) -> 'MarketData':
        """Add batch dimension if not present"""
        if len(self.prices.shape) == 2:
            return MarketData(
                prices=self.prices.unsqueeze(0),
                holdings=self.holdings.unsqueeze(0),
                metadata=self.metadata
            )
        return self
    
    def remove_batch_dim(self) -> 'MarketData':
        """Remove batch dimension if batch_size=1"""
        if len(self.prices.shape) == 3 and self.prices.shape[0] == 1:
            return MarketData(
                prices=self.prices.squeeze(0),
                holdings=self.holdings.squeeze(0),
                metadata=self.metadata
            )
        return self


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets"""
    
    def __init__(self, n_samples: Optional[int] = None):
        """
        Initialize base dataset.
        
        Args:
            n_samples: Number of samples. If None, will be determined dynamically.
        """
        self._n_samples = n_samples
    
    def __len__(self) -> int:
        """Return number of samples in dataset"""
        if self._n_samples is not None:
            return self._n_samples
        else:
            # For dynamic datasets, this should be overridden
            raise NotImplementedError("Dynamic datasets must override __len__")
    
    @abstractmethod
    def __getitem__(self, idx: int) -> MarketData:
        """Return MarketData instance for given index"""
        pass
    
    @property
    @abstractmethod
    def n_assets(self) -> int:
        """Number of assets in the dataset"""
        pass
    
    @property 
    def sequence_length(self) -> Optional[int]:
        """
        Sequence length of the data. 
        Returns None for datasets with variable sequence lengths.
        """
        return None
    
    def get_sequence_length(self, idx: int) -> int:
        """
        Get sequence length for a specific sample.
        For fixed-length datasets, this returns the same value for all indices.
        For variable-length datasets, this returns the actual length for the sample.
        """
        if self.sequence_length is not None:
            return self.sequence_length
        else:
            # Must be implemented by variable-length datasets
            raise NotImplementedError("Variable-length datasets must override get_sequence_length")
    
    @abstractmethod
    def get_sample_data(self) -> MarketData:
        """Get a sample data point for testing/validation"""
        pass