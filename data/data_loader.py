"""
Data loading and dataset management using the new interface architecture
"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Tuple, Optional, List, Dict, Union
from datetime import datetime, timedelta
import pandas as pd
import os

# New interface imports (no sys.path manipulation needed)
from .market_simulator import MarketSimulator
from .config import MarketConfig
from common.interfaces import DataResult, DatasetInterface, DataProviderInterface
from common.adapters import to_data_result, LegacyDataAdapter
from common.exceptions import DataProviderError, MissingDependencyError

# Data types for backward compatibility  
from .data_types import MarketData, BaseDataset

# Optional dependency management
class DeltaHedgeProvider:
    """Provider for delta hedge functionality with lazy loading"""
    
    def __init__(self):
        self._available = None
        self._modules = {}
    
    def is_available(self) -> bool:
        """Check if delta hedge modules are available"""
        if self._available is None:
            try:
                import delta_hedge
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def get_module(self, module_name: str):
        """Get delta hedge module with lazy loading"""
        if not self.is_available():
            raise MissingDependencyError(
                f"Delta hedge module '{module_name}' not available",
                context={'module': module_name}
            )
        
        if module_name not in self._modules:
            try:
                if module_name == 'weight_generator':
                    from delta_hedge.weight_generator import create_default_weight_generator
                    self._modules[module_name] = create_default_weight_generator
                elif module_name == 'data_loader':
                    from delta_hedge.data_loader import OptionsDataLoader
                    self._modules[module_name] = OptionsDataLoader
                elif module_name == 'config':
                    from delta_hedge.config import DeltaHedgeConfig
                    self._modules[module_name] = DeltaHedgeConfig
                else:
                    raise ImportError(f"Unknown delta hedge module: {module_name}")
            except ImportError as e:
                raise MissingDependencyError(
                    f"Failed to import delta hedge module '{module_name}'",
                    context={'module': module_name, 'error': str(e)}
                )
        
        return self._modules[module_name]

# Global delta hedge provider instance
_delta_provider = DeltaHedgeProvider()

def check_delta_availability() -> bool:
    """Check if delta hedge functionality is available"""
    return _delta_provider.is_available()

class TradingDataset(DatasetInterface):
    """Dataset for trading scenarios using new interface"""
    
    def __init__(self, simulator: MarketSimulator, n_samples: int):
        self.simulator = simulator
        self.n_samples = n_samples
        
        # Pre-generate all data
        self.prices, self.holdings = simulator.generate_batch(n_samples)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.prices = self.prices.cuda()
            self.holdings = self.holdings.cuda()
    
    def __len__(self) -> int:
        """Dataset size"""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> DataResult:
        """Get single sample using new DataResult format"""
        return DataResult(
            prices=self.prices[idx],
            holdings=self.holdings[idx],
            metadata={'dataset_type': 'trading', 'sample_idx': idx}
        )
    
    @property
    def n_assets(self) -> int:
        """Number of assets in the dataset"""
        return self.prices.shape[-1]
    
    @property
    def sequence_length(self) -> int:
        """Sequence length of the data"""
        return self.prices.shape[-2]
    
    def get_sample_data(self) -> DataResult:
        """Get a sample data point for testing/validation"""
        return self[0]
        
    # Backward compatibility method
    def get_legacy_format(self, idx: int) -> MarketData:
        """Get data in legacy MarketData format for backward compatibility"""
        data_result = self[idx]
        return MarketData(
            prices=data_result.prices,
            holdings=data_result.holdings,
            metadata=data_result.metadata
        )


class DeltaHedgeDataset(DatasetInterface):
    """Delta hedge dataset using new interface and dependency management"""
    
    def __init__(self, 
                 option_positions: Dict[str, float],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 sequence_length: int = 100):
        """
        Initialize delta hedge dataset with specific option positions.
        
        Args:
            option_positions: Specific option positions as dict 
                            {'3CN5/CALL_111.0': 1.0, '3CN5/PUT_111.5': -0.5}
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            sequence_length: Length of price/holding sequences
        """
        if not _delta_provider.is_available():
            raise MissingDependencyError(
                "Delta hedge functionality not available",
                context={'required_module': 'delta_hedge'}
            )
        
        self.start_date = start_date
        self.end_date = end_date
        self._sequence_length = sequence_length
        self.option_positions = option_positions
        
        # Initialize delta hedge components
        try:
            self._weight_generator = _delta_provider.get_module('weight_generator')
            self._options_loader = _delta_provider.get_module('data_loader')
        except MissingDependencyError as e:
            raise DataProviderError(f"Failed to initialize delta hedge components: {e}")
        
        # Load and prepare data
        self._load_data()
    
    def _load_data(self):
        """Load and prepare delta hedge data"""
        try:
            # This is a simplified version - in real implementation would 
            # use the actual delta hedge data loading logic
            # For now, create mock data to demonstrate the interface
            
            n_days = 50  # Mock number of days
            n_assets = len(self.option_positions)
            
            # Generate mock price and delta weight data
            self.prices = torch.randn(n_days, self.sequence_length, n_assets)
            self.delta_weights = torch.randn(n_days, self.sequence_length, n_assets)
            
            self.n_days = n_days
            
        except Exception as e:
            raise DataProviderError(f"Failed to load delta hedge data: {e}")
    
    def __len__(self) -> int:
        """Dataset size (number of days)"""
        return getattr(self, 'n_days', 0)
    
    def __getitem__(self, idx: int) -> DataResult:
        """Get single day's data"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self)}")
        
        return DataResult(
            prices=self.prices[idx],
            holdings=self.delta_weights[idx],  # Delta weights as holdings
            metadata={
                'dataset_type': 'delta_hedge',
                'day_idx': idx,
                'option_positions': self.option_positions,
                'sequence_length': self.sequence_length
            }
        )
    
    @property
    def n_assets(self) -> int:
        """Number of assets (option positions)"""
        return len(self.option_positions)
    
    @property 
    def sequence_length(self) -> Optional[int]:
        """Sequence length"""
        return getattr(self, '_sequence_length', None)
    
    def get_sample_data(self) -> DataResult:
        """Get sample data for testing"""
        if len(self) > 0:
            return self[0]
        else:
            raise DataProviderError("No data available in dataset")


def create_data_loader(config: MarketConfig, n_samples: int, batch_size: int) -> TorchDataLoader:
    """
    Create a data loader for traditional market simulation.
    
    Args:
        config: Market configuration
        n_samples: Number of samples to generate
        batch_size: Batch size for training
        
    Returns:
        DataLoader with market simulation data
    """
    from common.collate import data_result_collate_fn
    
    simulator = MarketSimulator(config)
    dataset = TradingDataset(simulator, n_samples)
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=data_result_collate_fn)


def create_delta_data_loader(batch_size: int,
                           option_positions: Dict[str, float],
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> TorchDataLoader:
    """
    Create a data loader for delta hedge data.
    
    Args:
        batch_size: Batch size for training
        option_positions: Specific option positions in format:
            {'3CN5/CALL_111.0': 1.0, '3CN5/PUT_111.5': -0.5}
            Keys are 'weekly_code/option_type_strike'
            Values are position quantities
        start_date: Start date for data (YYYY-MM-DD), if None uses all available
        end_date: End date for data (YYYY-MM-DD), if None uses all available
        
    Returns:
        DataLoader with delta hedge data (each sample is one day)
    """
    from common.collate import data_result_collate_fn
    
    dataset = DeltaHedgeDataset(
        option_positions=option_positions,
        start_date=start_date,
        end_date=end_date
    )
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          collate_fn=data_result_collate_fn)