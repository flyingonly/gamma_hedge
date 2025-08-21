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
from datetime import datetime

from common.interfaces import DataResult, DatasetInterface

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
                 min_sequences_per_option: int = 10):
        """
        Initialize dataset with precomputed Greeks data.
        
        Args:
            portfolio_positions: Dict mapping option keys to positions
                                e.g., {'3CN5/CALL_111.0': 1.0, '3IN5/PUT_107.0': -0.5}
            sequence_length: Length of each training sequence
            data_path: Path to preprocessed Greeks data
            min_sequences_per_option: Minimum sequences to generate per option
        """
        self.portfolio_positions = portfolio_positions
        self._sequence_length = sequence_length
        self.data_path = Path(data_path)
        self.min_sequences_per_option = min_sequences_per_option
        
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
        self.training_sequences = []
        
        # Find common time range across all options
        common_timestamps = self._find_common_timestamps(option_data)
        
        if len(common_timestamps) < self._sequence_length:
            logger.warning(f"Not enough common timestamps ({len(common_timestamps)}) for sequence length ({self._sequence_length})")
            if len(common_timestamps) == 0:
                return
            # Use available data with shorter sequences
            actual_seq_length = min(len(common_timestamps), self._sequence_length)
        else:
            actual_seq_length = self._sequence_length
        
        # Generate training sequences using sliding window
        max_sequences = max(
            len(common_timestamps) - actual_seq_length + 1,
            self.min_sequences_per_option
        )
        
        num_sequences = min(max_sequences, len(common_timestamps) - actual_seq_length + 1)
        
        for seq_idx in range(num_sequences):
            try:
                sequence_data = self._create_single_sequence(
                    option_data, common_timestamps, seq_idx, actual_seq_length
                )
                if sequence_data is not None:
                    self.training_sequences.append(sequence_data)
            except Exception as e:
                logger.warning(f"Failed to create sequence {seq_idx}: {e}")
                continue
        
        logger.info(f"Generated {len(self.training_sequences)} training sequences")
    
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
            'sequence_length': self._sequence_length
        }