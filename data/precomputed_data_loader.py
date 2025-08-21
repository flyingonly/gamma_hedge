"""
Precomputed Greeks Data Loader
============================

Loads and processes precomputed Greeks data for direct use in training.
Eliminates the need for separate integration and precomputation paths.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd

from common.interfaces import DataResult, DatasetInterface
from tools.vectorized_bs import (
    black_scholes_delta, 
    interpolate_implied_volatility,
    validate_bs_inputs
)

logger = logging.getLogger(__name__)


class PrecomputedGreeksDataset(DatasetInterface):
    """
    Dataset that directly uses precomputed Greeks data for training.
    Loads NPZ files from data/preprocessed_greeks/ and converts to training sequences.
    """
    
    def __init__(self, 
                 portfolio_positions: Dict[str, float],
                 sequence_length: int = 100,
                 data_path: str = "data/preprocessed_greeks",
                 min_sequences_per_option: int = 10,
                 underlying_dense_mode: bool = False,
                 underlying_data_path: str = "csv_process/underlying_npz",
                 # New time series parameters
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
            min_sequences_per_option: Minimum sequences to generate per option
            underlying_dense_mode: If True, use underlying data density for delta calculation
            underlying_data_path: Path to underlying NPZ data files
            data_split: 'all', 'train', 'val', 'test' - which split to use
            split_ratios: (train_ratio, val_ratio, test_ratio) for time-based splitting
            align_to_daily: If True, align sequences to trading day boundaries
            min_daily_sequences: Minimum sequences required per trading day
        """
        self.portfolio_positions = portfolio_positions
        self._sequence_length = sequence_length
        self.data_path = Path(data_path)
        self.min_sequences_per_option = min_sequences_per_option
        self.underlying_dense_mode = underlying_dense_mode
        self.underlying_data_path = Path(underlying_data_path)
        
        # New time series parameters
        self.data_split = data_split
        self.split_ratios = split_ratios
        self.align_to_daily = align_to_daily
        self.min_daily_sequences = min_daily_sequences
        
        # Storage for converted training sequences
        self.training_sequences: List[DataResult] = []
        
        # Load and convert precomputed data
        if self.underlying_dense_mode:
            self._load_underlying_dense_data()
        else:
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
        
        # Choose sequence creation strategy
        if self.align_to_daily:
            # Create daily-aligned sequences
            logger.info("Creating daily-aligned sequences for sparse mode")
            data_arrays = {
                'prices': underlying_prices,
                'deltas': portfolio_deltas
            }
            daily_groups = self._group_by_trading_date(common_timestamps, data_arrays)
            sequences = self._create_daily_aligned_sequences(daily_groups)
        else:
            # Use traditional sliding window approach
            logger.info("Creating sliding window sequences for sparse mode")
            sequences = self._create_sparse_sliding_window_sequences(
                option_data, common_timestamps, portfolio_deltas, underlying_prices
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
    
    def _create_sparse_sliding_window_sequences(self, 
                                              option_data: Dict[str, Dict],
                                              common_timestamps: np.ndarray,
                                              portfolio_deltas: np.ndarray,
                                              underlying_prices: np.ndarray) -> List[DataResult]:
        """Create sequences using traditional sliding window for sparse mode"""
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
        
        logger.debug(f"Creating {num_sequences} sparse sliding window sequences of length {actual_seq_length}")
        
        for seq_idx in range(num_sequences):
            try:
                # Extract sequence data
                start_idx = seq_idx
                end_idx = seq_idx + actual_seq_length
                
                seq_timestamps = common_timestamps[start_idx:end_idx]
                seq_prices = underlying_prices[start_idx:end_idx].reshape(-1, 1)
                seq_deltas = portfolio_deltas[start_idx:end_idx].reshape(-1, 1)
                
                # Create time features
                time_features = np.linspace(1.0, 0.0, actual_seq_length).reshape(-1, 1)
                
                # Convert to tensors
                prices_tensor = torch.from_numpy(seq_prices).float()
                holdings_tensor = torch.from_numpy(seq_deltas).float()
                time_features_tensor = torch.from_numpy(time_features).float()
                
                # Create DataResult
                data_result = DataResult(
                    prices=prices_tensor,
                    holdings=holdings_tensor,
                    time_features=time_features_tensor,
                    metadata={
                        'sequence_index': seq_idx,
                        'start_timestamp': seq_timestamps[0],
                        'end_timestamp': seq_timestamps[-1],
                        'mean_delta': np.mean(seq_deltas),
                        'price_range': (np.min(seq_prices), np.max(seq_prices)),
                        'is_daily_aligned': False,
                        'mode': 'sparse'
                    }
                )
                
                sequences.append(data_result)
                
            except Exception as e:
                logger.warning(f"Failed to create sparse sequence {seq_idx}: {e}")
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
    
    def _create_single_sequence(self, 
                              option_data: Dict[str, Dict], 
                              common_timestamps: np.ndarray,
                              start_idx: int,
                              seq_length: int) -> Optional[DataResult]:
        """Create a single training sequence from option data"""
        try:
            # Select timestamp range for this sequence
            seq_timestamps = common_timestamps[start_idx:start_idx + seq_length]
            
            # Initialize sequence arrays
            n_options = len(option_data)
            underlying_prices = np.zeros(seq_length)
            option_prices = np.zeros((seq_length, n_options))
            portfolio_deltas = np.zeros(seq_length)
            hedge_weights = np.zeros(seq_length)
            
            # Process each option in portfolio
            for opt_idx, (option_key, opt_data) in enumerate(option_data.items()):
                greeks = opt_data['greeks'] 
                position = opt_data['position']
                
                # Find indices for sequence timestamps in option data
                indices = self._find_timestamp_indices(greeks['timestamps'], seq_timestamps)
                
                if indices is None:
                    logger.warning(f"Could not find timestamp indices for {option_key}")
                    continue
                
                # Extract data for this sequence
                opt_underlying_prices = greeks['underlying_prices'][indices]
                opt_option_prices = greeks['option_prices'][indices]  
                opt_deltas = greeks['delta'][indices]
                
                # Use underlying prices from first option (they should be the same)
                if opt_idx == 0:
                    underlying_prices = opt_underlying_prices
                
                # Store option prices
                option_prices[:, opt_idx] = opt_option_prices
                
                # Accumulate portfolio delta (weighted by position)
                portfolio_deltas += position * opt_deltas
            
            # Calculate hedge weights (delta-neutral hedging)
            hedge_weights = -portfolio_deltas  # Negative of portfolio delta
            
            # Create time features (time remaining from 1.0 to 0.0)
            time_features = torch.linspace(1.0, 0.0, seq_length).unsqueeze(1)
            
            # Create DataResult with training format
            # Use underlying prices as 'prices' and hedge weights as 'holdings'
            sequence_result = DataResult(
                prices=torch.tensor(underlying_prices.reshape(seq_length, 1), dtype=torch.float32),
                holdings=torch.tensor(hedge_weights.reshape(seq_length, 1), dtype=torch.float32),
                time_features=time_features,
                metadata={
                    'dataset_type': 'precomputed_greeks',
                    'sequence_idx': start_idx,
                    'portfolio_positions': self.portfolio_positions.copy(),
                    'timestamps': seq_timestamps.tolist(),
                    'portfolio_deltas': portfolio_deltas.tolist(),
                    'option_prices': option_prices.tolist()
                }
            )
            
            return sequence_result
            
        except Exception as e:
            logger.error(f"Error creating sequence {start_idx}: {e}")
            return None
    
    def _find_timestamp_indices(self, 
                              option_timestamps: np.ndarray,
                              target_timestamps: np.ndarray) -> Optional[np.ndarray]:
        """Find indices of target timestamps in option timestamp array"""
        try:
            indices = []
            for target_ts in target_timestamps:
                # Find closest timestamp (exact match preferred)
                time_diffs = np.abs(option_timestamps - target_ts)
                closest_idx = np.argmin(time_diffs)
                
                # Only use if within reasonable tolerance (1 hour = 3600 seconds)
                if time_diffs[closest_idx] < 3600:
                    indices.append(closest_idx)
                else:
                    logger.warning(f"No close timestamp found for {target_ts}")
                    return None
            
            return np.array(indices)
            
        except Exception as e:
            logger.error(f"Error finding timestamp indices: {e}")
            return None
    
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
            'sequence_length': self._sequence_length,
            'underlying_dense_mode': self.underlying_dense_mode
        }
    
    def _load_underlying_dense_data(self):
        """
        Load underlying dense data and generate training sequences using interpolated IV
        and calculated deltas. This provides significantly more training data points.
        """
        logger.info("Loading underlying dense data mode...")
        
        # Load option data for IV interpolation
        option_data = {}
        underlying_codes = set()
        
        # Parse portfolio positions to extract underlying codes and option parameters
        for option_key, position in self.portfolio_positions.items():
            try:
                # Parse option key: '3CN5/CALL_111.0'
                weekly_code, option_spec = option_key.split('/')
                option_type, strike_str = option_spec.split('_')
                strike = float(strike_str)
                
                # Extract underlying code (first 3 characters typically)
                underlying_code = self._extract_underlying_code(weekly_code)
                underlying_codes.add(underlying_code)
                
                # Load option Greeks data for IV
                greeks_data = self._load_single_option_greeks(option_key)
                if greeks_data is not None:
                    option_data[option_key] = {
                        'greeks': greeks_data,
                        'position': position,
                        'underlying_code': underlying_code,
                        'option_type': option_type.upper(),
                        'strike': strike
                    }
                    logger.info(f"Loaded option data for {option_key}: {len(greeks_data['timestamps'])} points")
                else:
                    logger.warning(f"No option data found for {option_key}")
            except Exception as e:
                logger.error(f"Failed to parse option key {option_key}: {e}")
                continue
        
        if not option_data:
            raise ValueError("No valid option data loaded for portfolio")
        
        # Load underlying data for each underlying code
        underlying_data = {}
        for underlying_code in underlying_codes:
            try:
                underlying_file = self.underlying_data_path / f"{underlying_code}.npz"
                if underlying_file.exists():
                    data = np.load(underlying_file)
                    underlying_data[underlying_code] = {
                        'timestamps': data['timestamps'],
                        'prices': data['prices']
                    }
                    logger.info(f"Loaded underlying data for {underlying_code}: {len(data['timestamps'])} points")
                else:
                    logger.warning(f"Underlying data file not found: {underlying_file}")
            except Exception as e:
                logger.error(f"Failed to load underlying data for {underlying_code}: {e}")
                continue
        
        if not underlying_data:
            raise ValueError("No underlying data loaded")
        
        # Generate dense training sequences for each underlying
        for underlying_code, u_data in underlying_data.items():
            self._generate_dense_sequences_for_underlying(
                underlying_code, u_data, option_data
            )
        
        logger.info(f"Generated {len(self.training_sequences)} dense training sequences")
    
    def _extract_underlying_code(self, weekly_code: str) -> str:
        """Extract underlying code from weekly option code"""
        # Map weekly codes to underlying codes
        # This mapping should be based on your actual data structure
        weekly_to_underlying = {
            '3CN5': 'USU5',  # 10-year Treasury Note
            '3CQ5': 'USU5',
            '3IN5': 'FVU5',  # 5-year Treasury Note  
            '3IQ5': 'FVU5',
            '3MN5': 'TYU5',  # 10-year Treasury Note (alternative)
            '3MQ5': 'TYU5',
            '3WN5': 'TUU5',  # 2-year Treasury Note
            '3WQ5': 'TUU5'
        }
        
        return weekly_to_underlying.get(weekly_code, 'USU5')  # Default to USU5
    
    def _generate_dense_sequences_for_underlying(self,
                                               underlying_code: str,
                                               underlying_data: Dict,
                                               option_data: Dict):
        """
        Generate dense training sequences for a specific underlying asset
        
        Args:
            underlying_code: Code for underlying asset (e.g., 'USU5')
            underlying_data: Dict with 'timestamps' and 'prices' arrays
            option_data: Dict of all option data for interpolation
        """
        underlying_timestamps = underlying_data['timestamps']
        underlying_prices = underlying_data['prices']
        
        # Filter options that belong to this underlying
        relevant_options = {
            k: v for k, v in option_data.items() 
            if v['underlying_code'] == underlying_code
        }
        
        if not relevant_options:
            logger.warning(f"No options found for underlying {underlying_code}")
            return
        
        # For each option, interpolate IV and calculate delta
        portfolio_deltas = np.zeros(len(underlying_timestamps))
        
        for option_key, option_info in relevant_options.items():
            try:
                # Get option parameters
                strike = option_info['strike']
                option_type = option_info['option_type']
                position = option_info['position']
                greeks = option_info['greeks']
                
                # Interpolate implied volatility to underlying timestamps
                iv_interpolated = interpolate_implied_volatility(
                    option_timestamps=greeks['timestamps'],
                    option_iv=greeks['implied_volatility'],
                    target_timestamps=underlying_timestamps
                )
                
                # Calculate time to expiration (assuming constant for simplicity)
                # In practice, this should be calculated properly based on expiration dates
                time_to_expiry = self._calculate_time_to_expiry(underlying_timestamps)
                
                # Risk-free rate (assuming constant 5%)
                risk_free_rate = 0.05
                
                # Validate inputs before BS calculation
                if not validate_bs_inputs(underlying_prices, strike, time_to_expiry, 
                                        risk_free_rate, iv_interpolated, option_type):
                    logger.warning(f"Invalid BS inputs for {option_key}, skipping")
                    continue
                
                # Calculate delta using vectorized Black-Scholes
                option_deltas = black_scholes_delta(
                    S=underlying_prices,
                    K=strike,
                    T=time_to_expiry,
                    r=risk_free_rate,
                    sigma=iv_interpolated,
                    option_type=option_type.lower()
                )
                
                # Add to portfolio delta (weighted by position)
                portfolio_deltas += position * option_deltas
                
                logger.info(f"Calculated deltas for {option_key}: mean={np.mean(option_deltas):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to calculate deltas for {option_key}: {e}")
                continue
        
        # Generate training sequences using sliding window
        self._create_dense_training_sequences(
            underlying_timestamps, underlying_prices, portfolio_deltas, underlying_code
        )
    
    def _calculate_time_to_expiry(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Calculate time to expiration for each timestamp
        
        For simplicity, assuming weekly options with typical expiration patterns.
        In practice, this should use actual expiration dates.
        """
        # Assume options expire in 7 days on average (weekly options)
        # Convert to years (7 days / 365 days)
        return np.full(len(timestamps), 7.0 / 365.0)
    
    def _create_dense_training_sequences(self,
                                       timestamps: np.ndarray,
                                       prices: np.ndarray, 
                                       portfolio_deltas: np.ndarray,
                                       underlying_code: str):
        """
        Create training sequences from dense underlying data
        
        Args:
            timestamps: Array of underlying timestamps
            prices: Array of underlying prices
            portfolio_deltas: Array of calculated portfolio deltas
            underlying_code: Underlying asset code for logging
        """
        # Choose sequence creation strategy based on alignment setting
        if self.align_to_daily:
            # Create daily-aligned sequences
            logger.info(f"Creating daily-aligned sequences for {underlying_code}")
            data_arrays = {
                'prices': prices,
                'deltas': portfolio_deltas
            }
            daily_groups = self._group_by_trading_date(timestamps, data_arrays)
            sequences = self._create_daily_aligned_sequences(daily_groups)
        else:
            # Use traditional sliding window approach
            logger.info(f"Creating sliding window sequences for {underlying_code}")
            sequences = self._create_sliding_window_sequences(
                timestamps, prices, portfolio_deltas, underlying_code
            )
        
        # Apply time-based split if needed
        sequences = self._apply_time_based_split(sequences)
        
        # Add to training sequences
        self.training_sequences.extend(sequences)
        
        logger.info(f"Added {len(sequences)} sequences for {underlying_code} "
                   f"(align_to_daily={self.align_to_daily}, split={self.data_split})")
    
    def _create_sliding_window_sequences(self,
                                       timestamps: np.ndarray,
                                       prices: np.ndarray,
                                       portfolio_deltas: np.ndarray,
                                       underlying_code: str) -> List[DataResult]:
        """
        Create sequences using traditional sliding window approach (original implementation)
        """
        sequences = []
        n_points = len(timestamps)
        
        if n_points < self._sequence_length:
            logger.warning(f"Not enough data points ({n_points}) for sequence length ({self._sequence_length})")
            actual_seq_length = n_points
        else:
            actual_seq_length = self._sequence_length
        
        # Calculate number of sequences using sliding window
        num_sequences = max(1, n_points - actual_seq_length + 1)
        
        logger.debug(f"Creating {num_sequences} sliding window sequences of length {actual_seq_length}")
        
        for seq_idx in range(num_sequences):
            try:
                # Extract sequence data
                start_idx = seq_idx
                end_idx = seq_idx + actual_seq_length
                
                seq_timestamps = timestamps[start_idx:end_idx]
                seq_prices = prices[start_idx:end_idx].reshape(-1, 1)  # Shape: (seq_len, 1)
                seq_deltas = portfolio_deltas[start_idx:end_idx].reshape(-1, 1)  # Shape: (seq_len, 1)
                
                # Create holdings data - this represents the target hedge ratios
                # For delta hedging, holdings should equal delta values
                holdings = seq_deltas.copy()
                
                # Create time features (normalized time remaining)
                time_features = np.linspace(1.0, 0.0, actual_seq_length).reshape(-1, 1)
                
                # Convert to PyTorch tensors
                prices_tensor = torch.from_numpy(seq_prices).float()
                holdings_tensor = torch.from_numpy(holdings).float()
                time_features_tensor = torch.from_numpy(time_features).float()
                
                # Create DataResult object
                data_result = DataResult(
                    prices=prices_tensor,  # Shape: (seq_len, 1) - batch dimension added by DataLoader
                    holdings=holdings_tensor,  # Shape: (seq_len, 1) - batch dimension added by DataLoader
                    time_features=time_features_tensor,  # Shape: (seq_len, 1) - batch dimension added by DataLoader
                    metadata={
                        'underlying_code': underlying_code,
                        'sequence_index': seq_idx,
                        'start_timestamp': seq_timestamps[0],
                        'end_timestamp': seq_timestamps[-1],
                        'mean_delta': np.mean(seq_deltas),
                        'price_range': (np.min(seq_prices), np.max(seq_prices)),
                        'is_daily_aligned': False
                    }
                )
                
                sequences.append(data_result)
                
            except Exception as e:
                logger.warning(f"Failed to create sequence {seq_idx} for {underlying_code}: {e}")
                continue
        
        return sequences
    
    def _group_by_trading_date(self, timestamps: np.ndarray, data_arrays: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Group time series data by trading dates
        
        Args:
            timestamps: Array of timestamps (Unix timestamps)
            data_arrays: Dict of arrays to group by date (e.g., {'prices': prices, 'deltas': deltas})
            
        Returns:
            Dict mapping date strings to data for that date
        """
        if len(timestamps) == 0:
            return {}
        
        # Convert timestamps to datetime objects
        dt_timestamps = pd.to_datetime(timestamps, unit='s')
        
        # Group by date (ignoring time)
        dates = dt_timestamps.date
        unique_dates = sorted(set(dates))
        
        grouped_data = {}
        
        for date in unique_dates:
            # Find indices for this date
            date_mask = dates == date
            date_indices = np.where(date_mask)[0]
            
            if len(date_indices) == 0:
                continue
                
            date_str = date.isoformat()
            grouped_data[date_str] = {
                'timestamps': timestamps[date_indices],
                'indices': date_indices,
                'count': len(date_indices)
            }
            
            # Group all data arrays for this date
            for key, array in data_arrays.items():
                if len(array) == len(timestamps):
                    grouped_data[date_str][key] = array[date_indices]
                else:
                    logger.warning(f"Array {key} length mismatch: {len(array)} vs {len(timestamps)}")
        
        logger.info(f"Grouped data into {len(grouped_data)} trading dates")
        return grouped_data
    
    def _create_daily_aligned_sequences(self, daily_groups: Dict[str, Dict]) -> List[DataResult]:
        """
        Create training sequences aligned to trading day boundaries
        
        Args:
            daily_groups: Output from _group_by_trading_date()
            
        Returns:
            List of DataResult objects, one per trading day (if sufficient data)
        """
        sequences = []
        
        for date_str, day_data in daily_groups.items():
            try:
                day_count = day_data['count']
                
                # Skip days with insufficient data
                if day_count < self.min_daily_sequences:
                    logger.debug(f"Skipping {date_str}: only {day_count} data points (min: {self.min_daily_sequences})")
                    continue
                
                # Extract data arrays for this day
                timestamps = day_data['timestamps']
                prices = day_data.get('prices', np.array([]))
                deltas = day_data.get('deltas', np.array([]))
                
                if len(prices) == 0 or len(deltas) == 0:
                    logger.warning(f"Missing price or delta data for {date_str}")
                    continue
                
                # Create sequence for this trading day
                seq_prices = prices.reshape(-1, 1)  # Shape: (day_len, 1)
                seq_holdings = deltas.reshape(-1, 1)  # Delta hedging targets
                
                # Create time features (normalized time within the day)
                time_features = np.linspace(1.0, 0.0, day_count).reshape(-1, 1)
                
                # Convert to tensors
                prices_tensor = torch.from_numpy(seq_prices).float()
                holdings_tensor = torch.from_numpy(seq_holdings).float()
                time_features_tensor = torch.from_numpy(time_features).float()
                
                # Create DataResult object
                data_result = DataResult(
                    prices=prices_tensor,
                    holdings=holdings_tensor,
                    time_features=time_features_tensor,
                    metadata={
                        'trading_date': date_str,
                        'sequence_length': day_count,
                        'start_timestamp': timestamps[0],
                        'end_timestamp': timestamps[-1],
                        'mean_delta': np.mean(deltas),
                        'price_range': (np.min(prices), np.max(prices)),
                        'is_daily_aligned': True
                    }
                )
                
                sequences.append(data_result)
                logger.debug(f"Created daily sequence for {date_str}: {day_count} points")
                
            except Exception as e:
                logger.error(f"Failed to create daily sequence for {date_str}: {e}")
                continue
        
        logger.info(f"Created {len(sequences)} daily-aligned sequences")
        return sequences
    
    def _apply_time_based_split(self, sequences: List[DataResult]) -> List[DataResult]:
        """
        Apply time-based train/val/test split to sequences
        
        Args:
            sequences: List of DataResult objects sorted by time
            
        Returns:
            Filtered list based on self.data_split
        """
        if self.data_split == 'all' or len(sequences) == 0:
            return sequences
        
        # Ensure sequences are sorted by time
        sequences.sort(key=lambda x: x.metadata.get('start_timestamp', 0))
        
        # Calculate split boundaries
        n_total = len(sequences)
        train_ratio, val_ratio, test_ratio = self.split_ratios
        
        # Ensure ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
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