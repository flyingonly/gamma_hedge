"""
Unified Data Loading System for Gamma Hedge

This module provides a unified interface for all data loading operations,
replacing the fragmented data loading system with consistent interfaces
and standardized data formats.

Consolidates functionality from:
- data_loader.py
- precomputed_data_loader.py  
- options_loader.py
- market_simulator.py (data generation aspects)
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# New core interfaces
from core.interfaces import (
    TrainingData, DataProvider, TrainingPhase, 
    PortfolioPosition, create_portfolio_from_dict,
    BaseDataProcessor
)
from core.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedDataProcessor(BaseDataProcessor):
    """
    Unified data processor that handles all data loading scenarios:
    - Market simulation data
    - Precomputed Greeks data  
    - Options data with underlying assets
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_cache = {}
        
    def process_data(self, raw_data: Dict) -> TrainingData:
        """Process raw data into standard TrainingData format"""
        if 'prices' in raw_data and 'positions' in raw_data:
            # Already in standard format
            return TrainingData(
                prices=raw_data['prices'],
                positions=raw_data['positions'],
                metadata=raw_data.get('metadata', {})
            )
        else:
            # Need to convert from legacy format
            return self._convert_legacy_data(raw_data)
    
    def _convert_legacy_data(self, data: Dict) -> TrainingData:
        """Convert legacy data formats to TrainingData"""
        # Handle different legacy formats
        if 'holdings' in data:
            # Legacy format with 'holdings' instead of 'positions'
            return TrainingData(
                prices=data['prices'],
                positions=data['holdings'],
                metadata=data.get('metadata', {})
            )
        else:
            raise ValueError(f"Unknown data format: {data.keys()}")
    
    def get_data_statistics(self) -> Dict[str, any]:
        """Get statistics about processed data"""
        return {
            'processor_type': 'unified',
            'cache_size': len(self.data_cache),
            'config_mode': self.config.greeks.preprocessing_mode
        }


class MarketSimulationDataset(Dataset):
    """Dataset for synthetic market data using Geometric Brownian Motion"""
    
    def __init__(self, config: Config, n_samples: int, phase: TrainingPhase = TrainingPhase.TRAIN):
        self.config = config
        self.n_samples = n_samples
        self.phase = phase
        
        # Generate synthetic data
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic market data"""
        from .market_simulator import MarketSimulator
        
        simulator = MarketSimulator(self.config.market)
        self.prices, self.positions = simulator.generate_batch(self.n_samples)
        
        # Add time features
        seq_len = self.config.market.n_timesteps
        batch_size = self.n_samples
        n_assets = self.config.market.n_assets
        
        # Time remaining feature (1.0 at start, 0.0 at end)
        time_remaining = torch.linspace(1.0, 0.0, seq_len).unsqueeze(0).unsqueeze(-1)
        time_remaining = time_remaining.expand(batch_size, seq_len, n_assets)
        
        # Concatenate with prices to create expanded feature set
        self.prices = torch.cat([self.prices, time_remaining], dim=-1)
        
        logger.info(f"Generated {self.n_samples} synthetic sequences for {self.phase.value}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> TrainingData:
        return TrainingData(
            prices=self.prices[idx],
            positions=self.positions[idx], 
            metadata={
                'source': 'simulation',
                'phase': self.phase.value,
                'sample_idx': idx
            }
        )


class PrecomputedGreeksDataset(Dataset):
    """Dataset for precomputed Greeks data with portfolio positions"""
    
    def __init__(self, 
                 config: Config,
                 portfolio_positions: Dict[str, float],
                 phase: TrainingPhase = TrainingPhase.TRAIN,
                 sequence_length: Optional[int] = None):
        self.config = config
        self.portfolio_positions = portfolio_positions
        self.phase = phase
        self.sequence_length = sequence_length or config.delta_hedge.sequence_length
        
        # Convert portfolio format
        self.portfolio = create_portfolio_from_dict(portfolio_positions)
        
        # Load precomputed data
        self.data_sequences = []
        self._load_precomputed_data()
        
        logger.info(f"Loaded {len(self.data_sequences)} Greeks sequences for {self.phase.value}")
    
    def _load_precomputed_data(self):
        """Load precomputed Greeks data for portfolio options"""
        base_path = Path("data/preprocessed_greeks")
        
        for position in self.portfolio:
            try:
                # Parse option key: "3CN5/CALL_111.0"
                weekly_code, option_spec = position.option_key.split('/')
                option_type, strike = option_spec.split('_')
                
                # Load Greeks NPZ file
                file_path = base_path / weekly_code / f"{option_spec}_greeks.npz"
                
                if file_path.exists():
                    data = np.load(file_path)
                    
                    # Convert to training sequences
                    self._process_option_data(data, position)
                else:
                    logger.warning(f"Greeks data not found: {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to load data for {position.option_key}: {e}")
    
    def _process_option_data(self, data: np.ndarray, position: PortfolioPosition):
        """Process option Greeks data into training sequences"""
        # Extract relevant fields from NPZ data
        timestamps = data.get('timestamps', [])
        underlying_prices = data.get('underlying_prices', [])
        deltas = data.get('deltas', [])
        
        if len(timestamps) < self.sequence_length:
            logger.warning(f"Insufficient data for {position.option_key}: {len(timestamps)} < {self.sequence_length}")
            return
        
        # Create sequences
        n_sequences = len(timestamps) - self.sequence_length + 1
        
        for i in range(n_sequences):
            # Extract sequence
            seq_prices = torch.tensor(underlying_prices[i:i+self.sequence_length], dtype=torch.float32)
            seq_deltas = torch.tensor(deltas[i:i+self.sequence_length], dtype=torch.float32)
            
            # Calculate target positions (delta hedge)
            seq_positions = -seq_deltas * position.position_size  # Negative for hedging
            
            # Expand dimensions for consistency
            seq_prices = seq_prices.unsqueeze(-1)  # (seq_len, 1)
            seq_positions = seq_positions.unsqueeze(-1)  # (seq_len, 1)
            
            # Create TrainingData object
            training_data = TrainingData(
                prices=seq_prices,
                positions=seq_positions,
                metadata={
                    'option_key': position.option_key,
                    'position_size': position.position_size,
                    'sequence_start': i,
                    'source': 'precomputed_greeks'
                }
            )
            
            self.data_sequences.append(training_data)
    
    def __len__(self) -> int:
        return len(self.data_sequences)
    
    def __getitem__(self, idx: int) -> TrainingData:
        return self.data_sequences[idx]


class UnifiedDataProvider(DataProvider):
    """
    Unified data provider that implements the DataProvider protocol
    and handles all data loading scenarios
    """
    
    def __init__(self, config: Config, data_source: str = "auto"):
        self.config = config
        self.data_source = data_source
        self.processor = UnifiedDataProcessor(config)
        
        # Auto-detect data source if not specified
        if data_source == "auto":
            self.data_source = self._detect_data_source()
    
    def _detect_data_source(self) -> str:
        """Automatically detect best available data source"""
        # Check for precomputed Greeks data
        greeks_path = Path("data/preprocessed_greeks")
        if greeks_path.exists() and any(greeks_path.iterdir()):
            return "precomputed_greeks"
        
        # Check for raw options data
        options_path = Path("csv_process/weekly_options_data")
        if options_path.exists() and any(options_path.iterdir()):
            return "raw_options"
        
        # Fall back to simulation
        return "simulation"
    
    def get_training_data(self, 
                         batch_size: int,
                         sequence_length: int,
                         phase: TrainingPhase = TrainingPhase.TRAIN) -> TrainingData:
        """Get training data batch according to DataProvider protocol"""
        
        # Create appropriate dataset based on configuration and data source
        if self.config.delta_hedge.enable_delta and self.data_source in ["precomputed_greeks", "raw_options"]:
            # Use options data for delta hedging
            portfolio_positions = self._create_default_portfolio()
            dataset = PrecomputedGreeksDataset(
                config=self.config,
                portfolio_positions=portfolio_positions,
                phase=phase,
                sequence_length=sequence_length
            )
        else:
            # Use market simulation
            dataset = MarketSimulationDataset(
                config=self.config,
                n_samples=batch_size * 10,  # Generate more samples than needed
                phase=phase
            )
        
        # Create data loader
        data_loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(phase == TrainingPhase.TRAIN),
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Get first batch and convert to TrainingData format
        batch = next(iter(data_loader))
        
        if isinstance(batch, TrainingData):
            return batch
        else:
            # Batch is a list/tuple of TrainingData objects, need to combine
            return self._combine_batch(batch)
    
    def _create_default_portfolio(self) -> Dict[str, float]:
        """Create default portfolio based on configuration"""
        portfolio = {}
        
        # Use weekly codes from configuration
        for weekly_code in self.config.delta_hedge.weekly_codes:
            # Create a simple ATM call position
            portfolio[f"{weekly_code}/CALL_111.0"] = 1.0
            
        return portfolio
    
    def _combine_batch(self, batch_list: List[TrainingData]) -> TrainingData:
        """Combine a list of TrainingData objects into a single batched TrainingData"""
        # Stack all prices and positions
        prices = torch.stack([item.prices for item in batch_list])
        positions = torch.stack([item.positions for item in batch_list])
        
        # Combine metadata
        combined_metadata = {
            'batch_size': len(batch_list),
            'individual_metadata': [item.metadata for item in batch_list]
        }
        
        return TrainingData(
            prices=prices,
            positions=positions,
            metadata=combined_metadata
        )
    
    def get_data_info(self) -> Dict[str, any]:
        """Get information about available data"""
        return {
            'data_source': self.data_source,
            'delta_hedge_enabled': self.config.delta_hedge.enable_delta,
            'preprocessing_mode': self.config.greeks.preprocessing_mode,
            'weekly_codes': self.config.delta_hedge.weekly_codes,
            'processor_stats': self.processor.get_data_statistics()
        }


# Convenience functions for backward compatibility

def create_data_loader(config: Config, 
                      n_samples: int, 
                      batch_size: int,
                      phase: TrainingPhase = TrainingPhase.TRAIN) -> TorchDataLoader:
    """
    Create data loader for market simulation data
    
    Backward compatibility function that maintains the original API
    """
    dataset = MarketSimulationDataset(config, n_samples, phase)
    
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == TrainingPhase.TRAIN),
        num_workers=0
    )


def create_delta_data_loader(config: Config,
                           batch_size: int,
                           option_positions: Optional[Dict[str, float]] = None,
                           sequence_length: Optional[int] = None,
                           phase: TrainingPhase = TrainingPhase.TRAIN) -> TorchDataLoader:
    """
    Create data loader for delta hedging with options data
    
    Backward compatibility function that maintains the original API
    """
    if option_positions is None:
        option_positions = {}
        for weekly_code in config.delta_hedge.weekly_codes:
            option_positions[f"{weekly_code}/CALL_111.0"] = 1.0
    
    dataset = PrecomputedGreeksDataset(
        config=config,
        portfolio_positions=option_positions,
        phase=phase,
        sequence_length=sequence_length
    )
    
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == TrainingPhase.TRAIN),
        num_workers=0
    )


def create_unified_data_provider(config: Config, data_source: str = "auto") -> UnifiedDataProvider:
    """
    Create unified data provider - new recommended approach
    
    This is the preferred way to create data providers in the new architecture
    """
    return UnifiedDataProvider(config, data_source)


def check_delta_availability() -> bool:
    """Check if delta hedge functionality is available"""
    # Always available in unified system
    return True