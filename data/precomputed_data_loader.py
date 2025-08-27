"""
Precomputed Greeks Data Loader - Simplified
==========================================

Loads and processes precomputed Greeks data for direct use in training.
Simplified version without dense mode logic.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta, time
import pandas as pd

from core.interfaces import DataResult, DatasetInterface
from utils.historical_features import HistoricalFeatureCalculator
from utils.logger import get_logger

logger = get_logger(__name__)


class PrecomputedGreeksDataset(DatasetInterface):
    """
    Dataset that directly uses precomputed Greeks data for training.
    Loads NPZ files from data/preprocessed_greeks/ and converts to training sequences.
    Simplified version without dense mode logic.
    """
    
    def __init__(self, 
                 portfolio_positions: Dict[str, float],
                 sequence_length: int = 100,
                 data_path: str = "data/preprocessed_greeks",
                 preprocessing_mode: str = "sparse",
                 min_sequences_per_option: int = 10,
                 # Time series parameters
                 data_split: str = 'all',
                 split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                 align_to_daily: bool = False,
                 min_daily_sequences: int = 50):
        """
        Initialize dataset with precomputed Greeks data.
        
        Args:
            portfolio_positions: Dict mapping option keys to positions
                                e.g., {'3CN5/CALL_111.0': 1.0, '3IN5/PUT_107.0': -0.5}
            sequence_length: Length of each training sequence (ignored if align_to_daily=True)
            data_path: Path to preprocessed Greeks data
            preprocessing_mode: Which preprocessing mode data to use ('sparse', 'dense_interpolated', 'dense_daily_recalc')
            min_sequences_per_option: Minimum sequences to generate per option
            data_split: 'all', 'train', 'val', 'test' - which split to use
            split_ratios: (train_ratio, val_ratio, test_ratio) for time-based splitting
            align_to_daily: If True, align sequences to trading day boundaries
            min_daily_sequences: Minimum sequences required per trading day
        """
        self.portfolio_positions = portfolio_positions
        self._sequence_length = sequence_length
        self.data_path = Path(data_path) / preprocessing_mode  # Add mode subdirectory
        self.preprocessing_mode = preprocessing_mode
        self.min_sequences_per_option = min_sequences_per_option
        
        # Time series parameters
        self.data_split = data_split
        self.split_ratios = split_ratios
        self.align_to_daily = align_to_daily
        self.min_daily_sequences = min_daily_sequences
        
        # Historical feature calculator
        self.feature_calculator = HistoricalFeatureCalculator()
        
        # Storage for converted training sequences
        self.training_sequences: List[DataResult] = []
        
        # Load and convert precomputed data
        self._load_precomputed_data()
        
        logger.info(f"Loaded {len(self.training_sequences)} training sequences from precomputed Greeks")
    
    def _load_precomputed_data(self):
        """Load precomputed Greeks data and convert to training sequences"""
        option_data = {}
        
        # Load Greeks data for each option in portfolio
        for option_key, position in self.portfolio_positions.items():
            try:
                greeks_data = self._load_single_option_greeks(option_key)
                if greeks_data is not None:
                    option_data[option_key] = {
                        'greeks': greeks_data,
                        'position': position
                    }
                    logger.info(f"Loaded Greeks data for {option_key}: {len(greeks_data['timestamps'])} points")
                else:
                    logger.warning(f"No Greeks data found for {option_key}")
            except Exception as e:
                logger.error(f"Failed to load Greeks data for {option_key}: {e}")
                continue
        
        if not option_data:
            raise ValueError("No valid Greeks data loaded for any option in portfolio")
        
        # Convert to training sequences
        self._convert_to_training_sequences(option_data)
    
    def _load_single_option_greeks(self, option_key: str) -> Optional[Dict[str, np.ndarray]]:
        """Load precomputed Greeks data for a single option"""
        try:
            # Parse option key: '3CN5/CALL_111.0'
            weekly_code, option_spec = option_key.split('/')
            option_type, strike_str = option_spec.split('_')
            
            # Construct file path
            filename = f"{option_type}_{strike_str}_greeks.npz"
            file_path = self.data_path / weekly_code / filename
            
            if not file_path.exists():
                logger.warning(f"Greeks file not found: {file_path}")
                return None
            
            # Load NPZ file
            data = np.load(file_path)
            
            # Validate required fields
            required_fields = ['timestamps', 'underlying_prices', 'delta', 'option_prices']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field '{field}' in {file_path}")
                    return None
            
            # Convert to dictionary
            greeks_dict = {key: data[key] for key in data.keys()}
            
            return greeks_dict
            
        except Exception as e:
            logger.error(f"Error loading Greeks data for {option_key}: {e}")
            return None
    
    def _convert_to_training_sequences(self, option_data: Dict[str, Dict]):
        """Convert loaded Greeks data to training sequences"""
        # Find common time range across all options
        common_timestamps = self._find_common_timestamps(option_data)
        
        if len(common_timestamps) == 0:
            logger.error("No common timestamps found across options")
            return
        
        # Calculate portfolio delta for each timestamp
        portfolio_deltas, underlying_prices = self._calculate_portfolio_deltas(option_data, common_timestamps)
        
        # Extract delta features for enhanced time sensitivity
        delta_features = self._extract_delta_features(option_data, common_timestamps)
        
        # Choose sequence creation strategy
        if self.align_to_daily:
            # Create daily-aligned sequences
            logger.info("Creating daily-aligned sequences")
            data_arrays = {
                'prices': underlying_prices,
                'deltas': portfolio_deltas,
                'delta_features': delta_features
            }
            daily_groups = self._group_by_trading_date(common_timestamps, data_arrays)
            sequences = self._create_daily_aligned_sequences(daily_groups)
        else:
            # Use traditional sliding window approach
            logger.info("Creating sliding window sequences")
            sequences = self._create_sliding_window_sequences(
                common_timestamps, portfolio_deltas, underlying_prices, delta_features
            )
        
        # Apply time-based split
        sequences = self._apply_time_based_split(sequences)
        
        # Set training sequences
        self.training_sequences = sequences
        
        logger.info(f"Generated {len(self.training_sequences)} training sequences "
                   f"(align_to_daily={self.align_to_daily}, split={self.data_split})")
    
    def _calculate_portfolio_deltas(self, option_data: Dict[str, Dict], common_timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate portfolio delta for each common timestamp"""
        portfolio_deltas = np.zeros(len(common_timestamps))
        underlying_prices = np.zeros(len(common_timestamps))
        
        # Process each option in portfolio
        for option_key, data in option_data.items():
            position = data['position']
            greeks = data['greeks']
            
            # Find matching timestamps and interpolate if needed
            option_timestamps = greeks['timestamps']
            option_deltas = greeks['delta']
            option_prices = greeks['underlying_prices']
            
            # Interpolate to common timestamps
            interpolated_deltas = np.interp(common_timestamps, option_timestamps, option_deltas)
            interpolated_prices = np.interp(common_timestamps, option_timestamps, option_prices)
            
            # Add weighted contribution to portfolio
            portfolio_deltas += position * interpolated_deltas
            
            # Use prices from first option (they should be the same underlying)
            if np.all(underlying_prices == 0):
                underlying_prices = interpolated_prices
        
        return portfolio_deltas, underlying_prices
    
    def _extract_delta_features(self, option_data: Dict[str, Dict], common_timestamps: np.ndarray) -> np.ndarray:
        """Extract delta-based features for enhanced time sensitivity"""
        n_timestamps = len(common_timestamps)
        
        # Initialize delta features array
        # Features: [current_delta, delta_change, delta_acceleration]
        delta_features = np.zeros((n_timestamps, 3))
        
        # Calculate portfolio deltas first
        portfolio_deltas, _ = self._calculate_portfolio_deltas(option_data, common_timestamps)
        
        # Feature 1: Current portfolio delta (normalized by position size)
        total_position = sum(abs(data['position']) for data in option_data.values())
        normalized_delta = portfolio_deltas / max(total_position, 1e-6)
        delta_features[:, 0] = normalized_delta
        
        # Feature 2: Delta change rate (first derivative)
        if n_timestamps > 1:
            delta_change = np.gradient(portfolio_deltas)
            # Normalize by time interval (assuming timestamps are in seconds)
            time_diffs = np.gradient(common_timestamps)
            time_diffs = np.where(time_diffs > 0, time_diffs, 1.0)  # Avoid division by zero
            delta_change_rate = delta_change / time_diffs
            delta_features[:, 1] = delta_change_rate
        
        # Feature 3: Delta acceleration (second derivative)
        if n_timestamps > 2:
            delta_acceleration = np.gradient(delta_features[:, 1])
            delta_features[:, 2] = delta_acceleration
        
        return delta_features
    
    def _calculate_intraday_time_features(self, timestamps: np.ndarray) -> np.ndarray:
        """Calculate intraday time features based on time to market close"""
        time_features = np.zeros(len(timestamps))
        
        # Market close time (assuming US market: 4:00 PM ET = 16:00)
        market_close_time = time(16, 0)  # 4:00 PM
        
        for i, ts in enumerate(timestamps):
            try:
                # Convert timestamp to datetime
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts)
                else:
                    dt = pd.to_datetime(ts)
                
                # Get current time within the trading day
                current_time = dt.time()
                
                # Calculate seconds from market open (9:30 AM = 09:30) to close (4:00 PM = 16:00)
                market_open_seconds = 9.5 * 3600  # 9:30 AM in seconds
                market_close_seconds = 16 * 3600   # 4:00 PM in seconds
                total_trading_seconds = market_close_seconds - market_open_seconds
                
                # Current time in seconds from midnight
                current_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
                
                # Calculate time remaining to market close
                if current_seconds <= market_open_seconds:
                    # Before market open - full day remaining
                    time_remaining_ratio = 1.0
                elif current_seconds >= market_close_seconds:
                    # After market close - no time remaining
                    time_remaining_ratio = 0.0
                else:
                    # During market hours - calculate proportion remaining
                    elapsed_trading_seconds = current_seconds - market_open_seconds
                    time_remaining_ratio = 1.0 - (elapsed_trading_seconds / total_trading_seconds)
                    time_remaining_ratio = max(0.0, min(1.0, time_remaining_ratio))  # Clamp to [0,1]
                
                time_features[i] = time_remaining_ratio
                
            except Exception as e:
                logger.warning(f"Failed to parse timestamp {ts}: {e}")
                # Fallback to linear decay if timestamp parsing fails
                time_features[i] = 1.0 - (i / len(timestamps))
        
        return time_features.reshape(-1, 1)
    
    def _create_sliding_window_sequences(self, 
                                       common_timestamps: np.ndarray,
                                       portfolio_deltas: np.ndarray,
                                       underlying_prices: np.ndarray,
                                       delta_features: np.ndarray) -> List[DataResult]:
        """Create sequences using traditional sliding window"""
        sequences = []
        
        if len(common_timestamps) < self._sequence_length:
            logger.warning(f"Not enough common timestamps ({len(common_timestamps)}) for sequence length ({self._sequence_length})")
            if len(common_timestamps) == 0:
                return sequences
            actual_seq_length = len(common_timestamps)
        else:
            actual_seq_length = self._sequence_length
        
        # Generate training sequences using sliding window
        num_sequences = max(1, len(common_timestamps) - actual_seq_length + 1)
        
        # logger.debug(f"Creating {num_sequences} sliding window sequences of length {actual_seq_length}")
        
        for seq_idx in range(num_sequences):
            try:
                # Extract sequence data
                start_idx = seq_idx
                end_idx = seq_idx + actual_seq_length
                
                seq_timestamps = common_timestamps[start_idx:end_idx]
                seq_prices = underlying_prices[start_idx:end_idx].reshape(-1, 1)
                seq_deltas = portfolio_deltas[start_idx:end_idx].reshape(-1, 1)
                seq_delta_features = delta_features[start_idx:end_idx]  # Shape: (seq_len, 3)
                
                # Create intraday time features based on actual timestamps
                time_features = self._calculate_intraday_time_features(seq_timestamps)
                
                # Convert to tensors
                prices_tensor = torch.from_numpy(seq_prices).float()
                holdings_tensor = torch.from_numpy(seq_deltas).float()
                time_features_tensor = torch.from_numpy(time_features).float()
                delta_features_tensor = torch.from_numpy(seq_delta_features).float()
                
                # Calculate historical features
                # Create simulated execution history (assume partial execution based on delta changes)
                execution_history = self._estimate_execution_history(holdings_tensor)
                historical_features = self.feature_calculator.calculate_features(
                    prices_tensor, holdings_tensor, execution_history, delta_features_tensor
                )
                history_features_tensor = historical_features.to_tensor()
                
                
                # Create DataResult with delta features
                data_result = DataResult(
                    prices=prices_tensor,
                    holdings=holdings_tensor,
                    time_features=time_features_tensor,
                    history_features=history_features_tensor,
                    delta_features=delta_features_tensor,
                    metadata={
                        'sequence_index': seq_idx,
                        'start_timestamp': seq_timestamps[0],
                        'end_timestamp': seq_timestamps[-1],
                        'mean_delta': np.mean(seq_deltas),
                        'price_range': (np.min(seq_prices), np.max(seq_prices)),
                        'delta_stats': {
                            'mean_normalized_delta': np.mean(seq_delta_features[:, 0]),
                            'mean_delta_change_rate': np.mean(seq_delta_features[:, 1]),
                            'mean_delta_acceleration': np.mean(seq_delta_features[:, 2])
                        },
                        'is_daily_aligned': False
                    }
                )
                
                sequences.append(data_result)
                
            except Exception as e:
                logger.warning(f"Failed to create sequence {seq_idx}: {e}")
                continue
        
        return sequences
    
    def _find_common_timestamps(self, option_data: Dict[str, Dict]) -> np.ndarray:
        """Find timestamps common to all options in portfolio"""
        if not option_data:
            return np.array([])
        
        # Start with timestamps from first option
        first_option = list(option_data.keys())[0]
        common_timestamps = set(option_data[first_option]['greeks']['timestamps'])
        
        # Find intersection with other options
        for option_key, data in option_data.items():
            option_timestamps = set(data['greeks']['timestamps'])
            common_timestamps = common_timestamps.intersection(option_timestamps)
        
        # Convert to sorted array
        common_array = np.array(sorted(list(common_timestamps)))
        
        logger.info(f"Found {len(common_array)} common timestamps across {len(option_data)} options")
        return common_array
    
    def _group_by_trading_date(self, timestamps: np.ndarray, data_arrays: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Group data by trading dates"""
        daily_groups = {}
        
        for i, ts in enumerate(timestamps):
            try:
                # Convert timestamp to trading date
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts)
                else:
                    dt = pd.to_datetime(ts)
                
                date_key = dt.strftime('%Y-%m-%d')
                
                if date_key not in daily_groups:
                    daily_groups[date_key] = {
                        'indices': [],
                        'timestamps': [],
                        'data': {key: [] for key in data_arrays.keys()}
                    }
                
                daily_groups[date_key]['indices'].append(i)
                daily_groups[date_key]['timestamps'].append(ts)
                for key, array in data_arrays.items():
                    daily_groups[date_key]['data'][key].append(array[i])
                    
            except Exception as e:
                logger.warning(f"Failed to parse timestamp {ts}: {e}")
                continue
        
        return daily_groups
    
    def _create_daily_aligned_sequences(self, daily_groups: Dict[str, Dict]) -> List[DataResult]:
        """Create daily-aligned training sequences"""
        sequences = []
        
        for date_key, group_data in daily_groups.items():
            indices = group_data['indices']
            
            if len(indices) < self.min_daily_sequences:
                # logger.debug(f"Skipping date {date_key}: only {len(indices)} data points (min: {self.min_daily_sequences})")
                continue
            
            try:
                # Extract data for this date
                prices = np.array(group_data['data']['prices']).reshape(-1, 1)
                deltas = np.array(group_data['data']['deltas']).reshape(-1, 1)
                delta_features_data = np.array(group_data['data']['delta_features'])  # Shape: (seq_len, 3)
                timestamps_data = np.array(group_data['timestamps'])
                
                # Create intraday time features based on actual timestamps
                time_features = self._calculate_intraday_time_features(timestamps_data)
                seq_length = len(prices)
                
                # Convert to tensors
                prices_tensor = torch.from_numpy(prices).float()
                holdings_tensor = torch.from_numpy(deltas).float()
                time_features_tensor = torch.from_numpy(time_features).float()
                delta_features_tensor = torch.from_numpy(delta_features_data).float()
                
                # Calculate historical features
                execution_history = self._estimate_execution_history(holdings_tensor)
                historical_features = self.feature_calculator.calculate_features(
                    prices_tensor, holdings_tensor, execution_history, delta_features_tensor
                )
                history_features_tensor = historical_features.to_tensor()
                
                
                # Create DataResult with delta features
                data_result = DataResult(
                    prices=prices_tensor,
                    holdings=holdings_tensor,
                    time_features=time_features_tensor,
                    history_features=history_features_tensor,
                    delta_features=delta_features_tensor,
                    metadata={
                        'trading_date': date_key,
                        'sequence_length': seq_length,
                        'mean_delta': np.mean(deltas),
                        'price_range': (np.min(prices), np.max(prices)),
                        'delta_stats': {
                            'mean_normalized_delta': np.mean(delta_features_data[:, 0]),
                            'mean_delta_change_rate': np.mean(delta_features_data[:, 1]),
                            'mean_delta_acceleration': np.mean(delta_features_data[:, 2])
                        },
                        'is_daily_aligned': True
                    }
                )
                
                sequences.append(data_result)
                
            except Exception as e:
                logger.warning(f"Failed to create daily sequence for {date_key}: {e}")
                continue
        
        logger.info(f"Created {len(sequences)} daily-aligned sequences")
        return sequences
    
    def _apply_time_based_split(self, sequences: List[DataResult]) -> List[DataResult]:
        """Apply time-based data split"""
        if self.data_split == 'all':
            return sequences
        
        n_total = len(sequences)
        if n_total == 0:
            return sequences
        
        # Normalize split ratios
        train_ratio, val_ratio, test_ratio = self.split_ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        
        if total_ratio > 0:
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))
        
        # Apply split
        if self.data_split == 'train':
            split_sequences = sequences[:train_end]
        elif self.data_split == 'val':
            split_sequences = sequences[train_end:val_end]
        elif self.data_split == 'test':
            split_sequences = sequences[val_end:]
        else:
            raise ValueError(f"Invalid data_split: {self.data_split}")
        
        logger.info(f"Applied {self.data_split} split: {len(split_sequences)}/{n_total} sequences "
                   f"(ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f})")
        
        return split_sequences
    
    def _estimate_execution_history(self, holdings_tensor: torch.Tensor) -> torch.Tensor:
        """
        Estimate execution history from holdings changes.
        
        Since we don't have actual execution history, we simulate it based on
        how much the required holdings change between timesteps.
        
        Args:
            holdings_tensor: Required holdings sequence (seq_len, n_assets)
            
        Returns:
            Estimated binary execution history (seq_len, n_assets)
        """
        seq_len = holdings_tensor.shape[0]
        
        # Calculate holdings changes
        if seq_len <= 1:
            # For single timestep, assume execution occurred
            return torch.ones_like(holdings_tensor)
        
        holdings_changes = torch.diff(holdings_tensor, dim=0)
        
        # Estimate execution probability based on magnitude of required change
        # Larger changes are more likely to trigger execution
        change_magnitudes = torch.abs(holdings_changes)
        
        # Normalize by the maximum change to get probabilities
        max_change = torch.max(change_magnitudes) if torch.max(change_magnitudes) > 0 else 1.0
        execution_probs = change_magnitudes / max_change
        
        # Add some baseline execution probability and cap at 1.0
        execution_probs = torch.clamp(execution_probs + 0.2, 0.0, 1.0)
        
        # Sample binary execution decisions
        execution_history = torch.bernoulli(execution_probs)
        
        # Add initial timestep (assume no execution at start)
        initial_execution = torch.zeros(1, holdings_tensor.shape[1], dtype=holdings_tensor.dtype)
        execution_history = torch.cat([initial_execution, execution_history], dim=0)
        
        return execution_history
    
    def __len__(self) -> int:
        """Dataset size"""
        return len(self.training_sequences)
    
    def __getitem__(self, idx: int) -> DataResult:
        """Get single training sequence"""
        if idx >= len(self.training_sequences):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.training_sequences)}")
        
        return self.training_sequences[idx]
    
    @property
    def n_assets(self) -> int:
        """Number of assets (always 1 for underlying)"""
        return 1
    
    @property
    def sequence_length(self) -> int:
        """Sequence length"""
        return self._sequence_length
    
    def get_sample_data(self) -> DataResult:
        """Get sample data for shape validation"""
        if len(self.training_sequences) == 0:
            raise RuntimeError("No training sequences available")
        return self.training_sequences[0]
    
    def get_portfolio_info(self) -> Dict:
        """Get portfolio information"""
        return {
            'positions': self.portfolio_positions.copy(),
            'n_options': len(self.portfolio_positions),
            'n_sequences': len(self.training_sequences),
            'sequence_length': self._sequence_length
        }