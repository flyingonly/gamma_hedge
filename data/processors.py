"""
Unified Data Processing Utilities for Gamma Hedge

This module provides common data processing utilities used across
the data loading system, including time series processing, data
validation, and format conversions.
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path

from core.interfaces import TrainingData, DataMode
from core.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesProcessor:
    """Handles time series data processing and alignment"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def align_to_daily_boundaries(self, 
                                 timestamps: List[datetime], 
                                 data: np.ndarray) -> Tuple[List[np.ndarray], List[datetime]]:
        """
        Align time series data to daily trading boundaries
        
        Args:
            timestamps: List of timestamp objects
            data: Data array with first dimension matching timestamps
            
        Returns:
            Tuple of (daily_sequences, daily_dates)
        """
        if not timestamps or len(timestamps) == 0:
            return [], []
        
        # Convert to pandas for easier date handling
        df = pd.DataFrame({
            'timestamp': timestamps,
            'data_idx': range(len(timestamps))
        })
        
        # Extract date component
        df['date'] = df['timestamp'].dt.date
        
        # Group by date
        daily_groups = df.groupby('date')
        
        sequences = []
        dates = []
        
        for date, group in daily_groups:
            # Get indices for this day
            indices = group['data_idx'].values
            
            # Check minimum sequence length
            if len(indices) >= self.config.training.min_daily_sequences:
                # Extract data for this day
                daily_data = data[indices]
                sequences.append(daily_data)
                dates.append(date)
            else:
                logger.debug(f"Skipping date {date}: only {len(indices)} data points")
        
        logger.info(f"Aligned data to {len(sequences)} trading days")
        return sequences, dates
    
    def create_variable_length_sequences(self, 
                                       data_arrays: List[np.ndarray], 
                                       min_length: int = 10,
                                       max_length: Optional[int] = None) -> List[TrainingData]:
        """
        Create variable-length training sequences from daily data
        
        Args:
            data_arrays: List of daily data arrays
            min_length: Minimum sequence length
            max_length: Maximum sequence length (None for no limit)
            
        Returns:
            List of TrainingData objects
        """
        sequences = []
        
        for day_idx, daily_data in enumerate(data_arrays):
            if len(daily_data) < min_length:
                continue
                
            # Determine sequence length
            seq_len = len(daily_data)
            if max_length and seq_len > max_length:
                seq_len = max_length
            
            # Extract sequences (could create multiple overlapping sequences per day)
            for start_idx in range(0, len(daily_data) - seq_len + 1, seq_len // 2):
                end_idx = start_idx + seq_len
                
                if end_idx <= len(daily_data):
                    # Create sequence data
                    seq_data = daily_data[start_idx:end_idx]
                    
                    # Convert to TrainingData format
                    # Assuming data has structure: [prices, positions, ...]
                    if seq_data.shape[1] >= 2:
                        prices = torch.tensor(seq_data[:, 0:1], dtype=torch.float32)  # First column
                        positions = torch.tensor(seq_data[:, 1:2], dtype=torch.float32)  # Second column
                        
                        sequence = TrainingData(
                            prices=prices,
                            positions=positions,
                            metadata={
                                'day_index': day_idx,
                                'start_index': start_idx,
                                'sequence_length': seq_len,
                                'source': 'daily_aligned'
                            }
                        )
                        
                        sequences.append(sequence)
        
        logger.info(f"Created {len(sequences)} variable-length sequences")
        return sequences
    
    def apply_time_based_split(self, 
                              sequences: List[TrainingData],
                              split_ratios: Tuple[float, float, float]) -> Tuple[List, List, List]:
        """
        Apply time-based train/validation/test split
        
        Args:
            sequences: List of TrainingData objects
            split_ratios: (train_ratio, val_ratio, test_ratio)
            
        Returns:
            Tuple of (train_sequences, val_sequences, test_sequences)
        """
        if not sequences:
            return [], [], []
        
        train_ratio, val_ratio, test_ratio = split_ratios
        total_sequences = len(sequences)
        
        # Calculate split indices
        train_end = int(total_sequences * train_ratio)
        val_end = train_end + int(total_sequences * val_ratio)
        
        # Split sequences chronologically
        train_sequences = sequences[:train_end]
        val_sequences = sequences[train_end:val_end]
        test_sequences = sequences[val_end:]
        
        logger.info(f"Split {total_sequences} sequences: "
                   f"train={len(train_sequences)}, val={len(val_sequences)}, test={len(test_sequences)}")
        
        return train_sequences, val_sequences, test_sequences


class DataValidator:
    """Validates data quality and format consistency"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def validate_training_data(self, data: TrainingData) -> Dict[str, Any]:
        """
        Validate TrainingData object
        
        Returns:
            Validation results with any issues found
        """
        issues = []
        warnings = []
        
        # Check tensor shapes
        if data.prices.shape != data.positions.shape:
            issues.append(f"Shape mismatch: prices {data.prices.shape} vs positions {data.positions.shape}")
        
        # Check for NaN values
        if torch.isnan(data.prices).any():
            issues.append("NaN values found in prices")
        
        if torch.isnan(data.positions).any():
            issues.append("NaN values found in positions")
        
        # Check for infinite values
        if torch.isinf(data.prices).any():
            issues.append("Infinite values found in prices")
        
        if torch.isinf(data.positions).any():
            issues.append("Infinite values found in positions")
        
        # Check sequence length
        seq_len = data.sequence_length
        if seq_len < 10:
            warnings.append(f"Short sequence length: {seq_len}")
        elif seq_len > 10000:
            warnings.append(f"Very long sequence: {seq_len}")
        
        # Check value ranges
        price_range = (data.prices.min().item(), data.prices.max().item())
        if price_range[0] <= 0:
            warnings.append(f"Non-positive prices detected: min={price_range[0]}")
        
        position_range = (data.positions.min().item(), data.positions.max().item())
        if abs(position_range[1] - position_range[0]) > 1000:
            warnings.append(f"Large position range: {position_range}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'statistics': {
                'sequence_length': seq_len,
                'n_assets': data.n_assets,
                'price_range': price_range,
                'position_range': position_range
            }
        }
    
    def validate_npz_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate NPZ file structure and content
        
        Returns:
            Validation results for NPZ file
        """
        if not file_path.exists():
            return {
                'valid': False,
                'error': f'File not found: {file_path}',
                'file_path': str(file_path)
            }
        
        try:
            data = np.load(file_path)
            
            # Check required fields
            required_fields = ['timestamps', 'underlying_prices', 'deltas']
            missing_fields = [field for field in required_fields if field not in data.files]
            
            result = {
                'valid': len(missing_fields) == 0,
                'file_path': str(file_path),
                'fields': list(data.files),
                'missing_fields': missing_fields,
                'statistics': {}
            }
            
            # Add statistics for available fields
            for field in data.files:
                field_data = data[field]
                result['statistics'][field] = {
                    'shape': field_data.shape,
                    'dtype': str(field_data.dtype),
                    'size': field_data.size
                }
            
            data.close()
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Failed to load NPZ file: {e}',
                'file_path': str(file_path)
            }


class FormatConverter:
    """Converts between different data formats"""
    
    @staticmethod
    def legacy_to_training_data(prices: torch.Tensor, 
                               holdings: torch.Tensor,
                               metadata: Optional[Dict] = None) -> TrainingData:
        """Convert legacy format to TrainingData"""
        return TrainingData(
            prices=prices,
            positions=holdings,  # holdings -> positions
            metadata=metadata or {}
        )
    
    @staticmethod
    def training_data_to_legacy(data: TrainingData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert TrainingData to legacy format"""
        return data.prices, data.positions
    
    @staticmethod
    def numpy_to_training_data(prices_np: np.ndarray,
                              positions_np: np.ndarray,
                              metadata: Optional[Dict] = None) -> TrainingData:
        """Convert numpy arrays to TrainingData"""
        prices = torch.tensor(prices_np, dtype=torch.float32)
        positions = torch.tensor(positions_np, dtype=torch.float32)
        
        return TrainingData(
            prices=prices,
            positions=positions,
            metadata=metadata or {}
        )
    
    @staticmethod
    def training_data_to_numpy(data: TrainingData) -> Tuple[np.ndarray, np.ndarray]:
        """Convert TrainingData to numpy arrays"""
        prices_np = data.prices.detach().cpu().numpy()
        positions_np = data.positions.detach().cpu().numpy()
        return prices_np, positions_np


class DataNormalizer:
    """Handles data normalization and scaling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.price_stats = None
        self.position_stats = None
    
    def fit_normalization(self, training_data: List[TrainingData]):
        """Fit normalization parameters on training data"""
        all_prices = []
        all_positions = []
        
        for data in training_data:
            all_prices.append(data.prices.flatten())
            all_positions.append(data.positions.flatten())
        
        # Concatenate all data
        prices_tensor = torch.cat(all_prices)
        positions_tensor = torch.cat(all_positions)
        
        # Calculate statistics
        self.price_stats = {
            'mean': prices_tensor.mean(),
            'std': prices_tensor.std(),
            'min': prices_tensor.min(),
            'max': prices_tensor.max()
        }
        
        self.position_stats = {
            'mean': positions_tensor.mean(),
            'std': positions_tensor.std(),
            'min': positions_tensor.min(),
            'max': positions_tensor.max()
        }
        
        logger.info("Fitted normalization parameters")
    
    def normalize_training_data(self, data: TrainingData) -> TrainingData:
        """Apply normalization to TrainingData"""
        if self.price_stats is None or self.position_stats is None:
            raise ValueError("Normalization not fitted. Call fit_normalization first.")
        
        # Z-score normalization
        normalized_prices = (data.prices - self.price_stats['mean']) / self.price_stats['std']
        normalized_positions = (data.positions - self.position_stats['mean']) / self.position_stats['std']
        
        return TrainingData(
            prices=normalized_prices,
            positions=normalized_positions,
            metadata={**data.metadata, 'normalized': True}
        )
    
    def denormalize_training_data(self, data: TrainingData) -> TrainingData:
        """Reverse normalization on TrainingData"""
        if self.price_stats is None or self.position_stats is None:
            raise ValueError("Normalization not fitted.")
        
        # Reverse Z-score normalization
        denorm_prices = data.prices * self.price_stats['std'] + self.price_stats['mean']
        denorm_positions = data.positions * self.position_stats['std'] + self.position_stats['mean']
        
        return TrainingData(
            prices=denorm_prices,
            positions=denorm_positions,
            metadata={**data.metadata, 'normalized': False}
        )