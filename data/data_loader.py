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

def check_delta_availability() -> bool:
    """Check if delta hedge functionality is available"""
    # Delta hedge functionality is now integrated into unified data processing
    return True

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
        # Generate time features for this sequence
        seq_len = self.sequence_length
        time_features = torch.linspace(1.0, 0.0, seq_len).unsqueeze(1)  # time_remaining from 1 to 0
        
        return DataResult(
            prices=self.prices[idx],
            holdings=self.holdings[idx],
            metadata={'dataset_type': 'trading', 'sample_idx': idx},
            time_features=time_features
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
    from common.collate import data_result_collate_fn, variable_length_collate_fn
    
    simulator = MarketSimulator(config)
    dataset = TradingDataset(simulator, n_samples)
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=data_result_collate_fn)


def create_delta_data_loader(batch_size: int,
                           option_positions: Dict[str, float],
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           sequence_length: int = 100,
                           underlying_dense_mode: bool = False,
                           underlying_data_path: str = "csv_process/underlying_npz",
                           # New time series parameters
                           data_split: str = 'all',
                           split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                           align_to_daily: bool = False,
                           min_daily_sequences: int = 50) -> TorchDataLoader:
    """
    Create a data loader for delta hedge data using precomputed Greeks.
    
    Args:
        batch_size: Batch size for training
        option_positions: Specific option positions in format:
            {'3CN5/CALL_111.0': 1.0, '3CN5/PUT_111.5': -0.5}
            Keys are 'weekly_code/option_type_strike'
            Values are position quantities
        start_date: Start date for data (YYYY-MM-DD) - currently unused, reserved for future
        end_date: End date for data (YYYY-MM-DD) - currently unused, reserved for future  
        sequence_length: Length of training sequences (ignored if align_to_daily=True)
        underlying_dense_mode: If True, use underlying data density with interpolated IV
        underlying_data_path: Path to underlying NPZ data files
        data_split: Which data split to use ('all', 'train', 'val', 'test')
        split_ratios: (train_ratio, val_ratio, test_ratio) for time-based splitting
        align_to_daily: If True, align sequences to trading day boundaries
        min_daily_sequences: Minimum data points required per trading day
        
    Returns:
        DataLoader with precomputed Greeks-based training data
    """
    from common.collate import data_result_collate_fn, variable_length_collate_fn
    from .precomputed_data_loader import PrecomputedGreeksDataset
    
    dataset = PrecomputedGreeksDataset(
        portfolio_positions=option_positions,
        sequence_length=sequence_length,
        underlying_dense_mode=underlying_dense_mode,
        underlying_data_path=underlying_data_path,
        data_split=data_split,
        split_ratios=split_ratios,
        align_to_daily=align_to_daily,
        min_daily_sequences=min_daily_sequences
    )
    # Choose collate function based on alignment mode
    collate_fn = variable_length_collate_fn if align_to_daily else data_result_collate_fn
    
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          collate_fn=collate_fn)


def create_underlying_dense_data_loader(batch_size: int,
                                       option_positions: Dict[str, float],
                                       sequence_length: int = 1000,
                                       underlying_data_path: str = "csv_process/underlying_npz",
                                       # New time series parameters
                                       data_split: str = 'all',
                                       split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                                       align_to_daily: bool = False,
                                       min_daily_sequences: int = 50) -> TorchDataLoader:
    """
    Create a data loader using underlying dense mode for maximum training data.
    This is a convenience function that automatically enables underlying_dense_mode.
    
    Args:
        batch_size: Batch size for training
        option_positions: Specific option positions in format:
            {'3CN5/CALL_111.0': 1.0, '3CN5/PUT_111.5': -0.5}
        sequence_length: Length of training sequences (default 1000 for dense data)
        underlying_data_path: Path to underlying NPZ data files
        data_split: Which data split to use ('all', 'train', 'val', 'test')
        split_ratios: (train_ratio, val_ratio, test_ratio) for time-based splitting
        align_to_daily: If True, align sequences to trading day boundaries
        min_daily_sequences: Minimum data points required per trading day
        
    Returns:
        DataLoader with underlying dense training data
    """
    return create_delta_data_loader(
        batch_size=batch_size,
        option_positions=option_positions,
        sequence_length=sequence_length,
        underlying_dense_mode=True,
        underlying_data_path=underlying_data_path,
        data_split=data_split,
        split_ratios=split_ratios,
        align_to_daily=align_to_daily,
        min_daily_sequences=min_daily_sequences
    )