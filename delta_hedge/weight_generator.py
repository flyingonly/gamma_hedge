"""
Weight Generator Module
======================

Generates delta hedging weights in format compatible with policy learning framework.
Provides interface between delta calculations and existing training system.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path for imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_simulator import MarketConfig
from .config import DeltaHedgeConfig
from .data_loader import OptionsDataLoader
from .delta_calculator import DeltaCalculator

# Setup logging
logger = logging.getLogger(__name__)


class WeightGenerator:
    """
    Generates delta hedging weights for policy learning integration.
    
    Provides:
    - Market data simulation with delta hedging weights
    - PyTorch tensor outputs compatible with existing training framework
    - Batch generation for training
    - Integration with existing market simulator interface
    """
    
    def __init__(self, config: DeltaHedgeConfig):
        """
        Initialize weight generator.
        
        Args:
            config: Delta hedging configuration
        """
        self.config = config
        self.data_loader = OptionsDataLoader(config)
        self.delta_calculator = DeltaCalculator(config)
        
        # Cache for loaded market data
        self.cached_market_data = None
        self.cached_options_data = None
        
        logger.info("Weight generator initialized")
    
    def generate_delta_weights_batch(self, 
                                   underlying_codes: List[str],
                                   weekly_codes: List[str],
                                   batch_size: int,
                                   sequence_length: int,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> Dict[str, torch.Tensor]:
        """
        Generate batch of delta hedging weights for training.
        
        Args:
            underlying_codes: List of underlying asset codes
            weekly_codes: List of weekly option codes  
            batch_size: Number of sequences in batch
            sequence_length: Length of each sequence
            start_date: Optional start date for data
            end_date: Optional end date for data
            
        Returns:
            Dictionary with tensors compatible with policy learning framework
        """
        logger.info(f"Generating delta weights batch: {batch_size} sequences of length {sequence_length}")
        
        # Load market data if not cached
        if self.cached_market_data is None:
            self.cached_market_data = self.data_loader.create_market_dataset(
                underlying_codes=underlying_codes,
                weekly_codes=weekly_codes,
                start_date=start_date,
                end_date=end_date
            )
        
        # Generate sequences
        batch_data = {
            'prices': [],
            'holdings': [],
            'delta_weights': [],
            'delta_info': [],
            'risk_metrics': []
        }
        
        for batch_idx in range(batch_size):
            sequence_data = self._generate_single_sequence(
                underlying_codes=underlying_codes,
                weekly_codes=weekly_codes,
                sequence_length=sequence_length
            )
            
            batch_data['prices'].append(sequence_data['prices'])
            batch_data['holdings'].append(sequence_data['holdings'])
            batch_data['delta_weights'].append(sequence_data['delta_weights'])
            batch_data['delta_info'].append(sequence_data['delta_info'])
            batch_data['risk_metrics'].append(sequence_data['risk_metrics'])
        
        # Convert to tensors
        tensor_data = {
            'prices': torch.stack(batch_data['prices']),
            'holdings': torch.stack(batch_data['holdings']),
            'delta_weights': torch.stack(batch_data['delta_weights']),
            'delta_info': torch.stack(batch_data['delta_info']),
            'risk_metrics': torch.stack(batch_data['risk_metrics'])
        }
        
        logger.info(f"Generated batch with shapes: {self._get_tensor_shapes(tensor_data)}")
        return tensor_data
    
    def _generate_single_sequence(self, underlying_codes: List[str],
                                weekly_codes: List[str],
                                sequence_length: int) -> Dict[str, torch.Tensor]:
        """Generate a single sequence of market data with delta weights"""
        
        n_assets = len(underlying_codes)
        
        # Generate underlying price paths (simplified GBM)
        prices = self._simulate_price_paths(underlying_codes, sequence_length)
        
        # Generate holdings schedule (placeholder - in practice this could be optimized)
        holdings = self._generate_holdings_schedule(underlying_codes, sequence_length)
        
        # Create sample portfolio positions for options
        portfolio_positions = self._create_sample_portfolio(weekly_codes)
        
        # Generate options time series data
        options_timeseries = self._generate_options_timeseries(
            weekly_codes, prices, sequence_length
        )
        
        # Get contract information for the first weekly code (for expiry checking)
        # TODO: This assumes all options in the portfolio have similar expiry
        # In practice, we might need to handle multiple contracts with different expiries
        contract_info = None
        if weekly_codes:
            contract_info = self.data_loader.get_option_contract_info(weekly_codes[0])
            if contract_info:
                logger.info(f"Using contract info from {weekly_codes[0]}: expiry={contract_info.get('expiry_date')}")
        
        # Calculate delta weights
        timestamps = np.arange(sequence_length)
        delta_results = self.delta_calculator.calculate_dynamic_hedge_weights(
            underlying_prices=prices[:, 0],  # Use first asset price
            options_timeseries=options_timeseries,
            portfolio_positions=portfolio_positions,
            timestamps=timestamps,
            contract_info=contract_info
        )
        
        # Create delta weights tensor (matching holdings format)
        delta_weights = np.zeros((sequence_length, n_assets))
        delta_weights[:, 0] = delta_results['hedge_ratios']  # Apply to first asset
        
        # Create delta info tensor (additional information for model)
        delta_info = np.stack([
            delta_results['portfolio_deltas'],
            delta_results['hedging_costs'],
            delta_results['rebalance_flags'].astype(float)
        ], axis=1)  # Shape: (sequence_length, 3)
        
        # Create risk metrics tensor
        risk_metrics = np.zeros((sequence_length, 4))  # 4 risk metrics
        risk_metrics[:, 0] = delta_results['portfolio_deltas']  # Portfolio delta
        risk_metrics[:, 1] = np.cumsum(delta_results['hedging_costs'])  # Cumulative cost
        
        # Calculate additional risk metrics
        if len(delta_results['portfolio_deltas']) > 1:
            rolling_vol = self._calculate_rolling_volatility(delta_results['portfolio_deltas'])
            risk_metrics[:, 2] = rolling_vol
            
            hedge_effectiveness = delta_results.get('hedge_effectiveness', {})
            risk_metrics[:, 3] = hedge_effectiveness.get('effectiveness', 0.0)
        
        return {
            'prices': torch.tensor(prices, dtype=torch.float32),
            'holdings': torch.tensor(holdings, dtype=torch.float32),
            'delta_weights': torch.tensor(delta_weights, dtype=torch.float32),
            'delta_info': torch.tensor(delta_info, dtype=torch.float32),
            'risk_metrics': torch.tensor(risk_metrics, dtype=torch.float32)
        }
    
    def _simulate_price_paths(self, underlying_codes: List[str], 
                            sequence_length: int) -> np.ndarray:
        """Simulate price paths using GBM"""
        n_assets = len(underlying_codes)
        dt = 1/252  # Daily time step
        
        # Parameters for GBM
        initial_prices = np.array([100.0] * n_assets)  # Start at 100
        drifts = np.array([0.05] * n_assets)  # 5% annual drift
        volatilities = np.array([0.2] * n_assets)  # 20% annual volatility
        
        # Generate random shocks
        random_shocks = np.random.randn(sequence_length, n_assets)
        
        # Initialize price paths
        prices = np.zeros((sequence_length, n_assets))
        prices[0] = initial_prices
        
        # Generate GBM paths
        for t in range(1, sequence_length):
            dW = np.sqrt(dt) * random_shocks[t]
            prices[t] = prices[t-1] * (
                1 + drifts * dt + volatilities * dW
            )
        
        return prices
    
    def _generate_holdings_schedule(self, underlying_codes: List[str],
                                  sequence_length: int) -> np.ndarray:
        """Generate holdings schedule (similar to existing market simulator)"""
        n_assets = len(underlying_codes)
        
        # Generate target holdings path
        initial_holding = 5.0
        target_holding = 7.0
        
        holdings = np.zeros((sequence_length, n_assets))
        
        for i in range(sequence_length):
            # Linear interpolation from initial to target
            alpha = i / (sequence_length - 1) if sequence_length > 1 else 0
            base_holding = (1 - alpha) * initial_holding + alpha * target_holding
            
            # Add some noise
            noise = np.random.randn(n_assets) * 0.1
            holdings[i] = base_holding + noise
        
        return holdings
    
    def _create_sample_portfolio(self, weekly_codes: List[str]) -> Dict[str, float]:
        """Create sample portfolio positions for options"""
        if not weekly_codes:
            return {}
        
        # Create simple portfolio with random positions
        portfolio = {}
        
        for weekly_code in weekly_codes:
            # Get available options for this weekly code
            options_data = self.data_loader.load_options_data(weekly_code)
            
            # Select a few options randomly
            option_ids = list(options_data.keys())[:3]  # Take first 3 options
            
            for option_id in option_ids:
                # Random position size
                position_size = np.random.choice([-2, -1, 1, 2]) * np.random.uniform(0.5, 2.0)
                portfolio[f"{weekly_code}_{option_id}"] = position_size
        
        return portfolio
    
    def _generate_options_timeseries(self, weekly_codes: List[str],
                                   underlying_prices: np.ndarray,
                                   sequence_length: int) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate options time series data"""
        options_timeseries = {}
        
        for weekly_code in weekly_codes:
            # Load options data
            options_data = self.data_loader.load_options_data(weekly_code)
            
            for option_id, option_df in options_data.items():
                if len(option_df) == 0:
                    continue
                
                # Get option parameters
                strike = option_df['strike'].iloc[0]
                option_type = option_df['option_type'].iloc[0]
                
                # Generate synthetic option prices based on underlying movement
                option_prices = self._calculate_synthetic_option_prices(
                    underlying_prices[:, 0], strike, option_type, sequence_length
                )
                
                # Create timeseries data
                full_option_id = f"{weekly_code}_{option_id}"
                options_timeseries[full_option_id] = {
                    'market_prices': option_prices,
                    'strikes': np.full(sequence_length, strike),
                    'option_types': [option_type] * sequence_length,
                    'implied_volatilities': np.full(sequence_length, 0.2)  # Default 20%
                }
        
        return options_timeseries
    
    def _calculate_synthetic_option_prices(self, underlying_prices: np.ndarray,
                                         strike: float, option_type: str,
                                         sequence_length: int) -> np.ndarray:
        """Calculate synthetic option prices using simplified BS"""
        option_prices = np.zeros(sequence_length)
        
        for i, S in enumerate(underlying_prices):
            # Simplified option pricing
            moneyness = S / strike
            
            if option_type.upper() == 'CALL':
                # Simplified call price
                intrinsic = max(0, S - strike)
                time_value = max(0.01, 0.1 * strike * (30/365)**0.5)  # Simplified time value
                option_prices[i] = intrinsic + time_value
            else:  # PUT
                # Simplified put price
                intrinsic = max(0, strike - S)
                time_value = max(0.01, 0.1 * strike * (30/365)**0.5)  # Simplified time value
                option_prices[i] = intrinsic + time_value
        
        return option_prices
    
    def _calculate_rolling_volatility(self, values: np.ndarray, window: int = 5) -> np.ndarray:
        """Calculate rolling volatility"""
        if len(values) < window:
            return np.full(len(values), 0.0)
        
        rolling_vol = np.zeros(len(values))
        
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i+1]
            
            if len(window_values) > 1:
                rolling_vol[i] = np.std(window_values)
            else:
                rolling_vol[i] = 0.0
        
        return rolling_vol
    
    def _get_tensor_shapes(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, tuple]:
        """Get shapes of tensors for logging"""
        return {key: tuple(tensor.shape) for key, tensor in tensor_dict.items()}
    
    def create_market_simulator_compatible_data(self, 
                                              underlying_codes: List[str],
                                              weekly_codes: List[str],
                                              n_paths: int = 100,
                                              n_timesteps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create data compatible with existing MarketSimulator interface.
        
        Args:
            underlying_codes: List of underlying codes
            weekly_codes: List of weekly option codes
            n_paths: Number of paths to generate
            n_timesteps: Number of timesteps per path
            
        Returns:
            Tuple of (prices, holdings) tensors compatible with existing training
        """
        logger.info(f"Creating market simulator compatible data: {n_paths} paths, {n_timesteps} timesteps")
        
        # Generate batch data
        batch_data = self.generate_delta_weights_batch(
            underlying_codes=underlying_codes,
            weekly_codes=weekly_codes,
            batch_size=n_paths,
            sequence_length=n_timesteps
        )
        
        # Extract prices and holdings (standard format)
        prices = batch_data['prices']  # Shape: (n_paths, n_timesteps, n_assets)
        holdings = batch_data['holdings']  # Shape: (n_paths, n_timesteps, n_assets)
        
        # Optionally modify holdings to include delta hedging information
        if self.config.policy_integration.extend_state_space:
            # Add delta weights as additional "holdings" information
            delta_weights = batch_data['delta_weights']
            
            # Concatenate holdings and delta weights
            extended_holdings = torch.cat([holdings, delta_weights], dim=-1)
            return prices, extended_holdings
        
        return prices, holdings
    
    def get_feature_dimensions(self, underlying_codes: List[str],
                             weekly_codes: List[str]) -> Dict[str, int]:
        """Get feature dimensions for model configuration"""
        n_assets = len(underlying_codes)
        n_options = sum(len(self.data_loader.load_options_data(code)) for code in weekly_codes)
        
        dimensions = {
            'n_assets': n_assets,
            'n_options': n_options,
            'price_dim': n_assets,
            'holdings_dim': n_assets,
            'delta_weights_dim': n_assets,
            'delta_info_dim': 3,  # portfolio_delta, hedging_cost, rebalance_flag
            'risk_metrics_dim': 4  # portfolio_delta, cumulative_cost, volatility, effectiveness
        }
        
        # Extended state space dimension
        if self.config.policy_integration.extend_state_space:
            state_dim = n_assets * 3 + 3 + 4  # prices + holdings + delta_weights + delta_info + risk_metrics
            dimensions['extended_state_dim'] = state_dim
        
        return dimensions
    
    def apply_weight_scaling(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply scaling to weights according to configuration"""
        scaling_method = self.config.weight_generation.weight_scaling_method
        target_range = self.config.weight_generation.target_weight_range
        
        if scaling_method == "normalize":
            # Normalize to zero mean, unit variance
            weights_normalized = (weights - weights.mean()) / (weights.std() + 1e-8)
            
            # Scale to target range
            min_target, max_target = target_range
            weights_scaled = weights_normalized * (max_target - min_target) / 4  # Approximately 95% within range
            
        elif scaling_method == "standardize":
            # Z-score standardization
            weights_scaled = (weights - weights.mean()) / (weights.std() + 1e-8)
            
        elif scaling_method == "minmax":
            # Min-max scaling to target range
            min_target, max_target = target_range
            weights_min = weights.min()
            weights_max = weights.max()
            
            if weights_max > weights_min:
                weights_scaled = (weights - weights_min) / (weights_max - weights_min)
                weights_scaled = weights_scaled * (max_target - min_target) + min_target
            else:
                weights_scaled = torch.full_like(weights, (min_target + max_target) / 2)
        
        else:
            weights_scaled = weights
        
        # Apply noise if configured
        if self.config.weight_generation.add_weight_noise:
            noise_std = self.config.weight_generation.weight_noise_std
            noise = torch.randn_like(weights_scaled) * noise_std
            weights_scaled += noise
        
        return weights_scaled
    
    def save_generated_data(self, data: Dict[str, torch.Tensor], 
                          filename: str = "delta_hedge_data.npz"):
        """Save generated data to file"""
        save_path = f"./delta_hedge/{filename}"
        
        # Convert tensors to numpy for saving
        numpy_data = {key: tensor.numpy() for key, tensor in data.items()}
        
        np.savez_compressed(save_path, **numpy_data)
        logger.info(f"Saved generated data to {save_path}")
    
    def load_saved_data(self, filename: str = "delta_hedge_data.npz") -> Dict[str, torch.Tensor]:
        """Load previously saved data"""
        load_path = f"./delta_hedge/{filename}"
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Data file not found: {load_path}")
        
        # Load numpy data and convert to tensors
        numpy_data = np.load(load_path)
        tensor_data = {key: torch.tensor(array) for key, array in numpy_data.items()}
        
        logger.info(f"Loaded data from {load_path}")
        return tensor_data


def create_default_weight_generator(underlying_codes: Optional[List[str]] = None,
                                  weekly_codes: Optional[List[str]] = None) -> WeightGenerator:
    """Create weight generator with default configuration"""
    config = DeltaHedgeConfig()
    
    if underlying_codes:
        config.data.available_underlyings = underlying_codes
    if weekly_codes:
        # Set available weekly codes in config if needed
        pass
    
    return WeightGenerator(config)


def test_weight_generation():
    """Test function for weight generation"""
    # Test configuration
    underlying_codes = ['USU5', 'FVU5']
    weekly_codes = ['3CN5', '3IN5']
    
    # Create generator
    generator = create_default_weight_generator(underlying_codes, weekly_codes)
    
    # Generate test batch
    batch_data = generator.generate_delta_weights_batch(
        underlying_codes=underlying_codes,
        weekly_codes=weekly_codes,
        batch_size=4,
        sequence_length=10
    )
    
    print("Generated batch with shapes:")
    for key, tensor in batch_data.items():
        print(f"  {key}: {tensor.shape}")
    
    # Test market simulator compatibility
    prices, holdings = generator.create_market_simulator_compatible_data(
        underlying_codes=underlying_codes,
        weekly_codes=weekly_codes,
        n_paths=10,
        n_timesteps=15
    )
    
    print(f"\nMarket simulator compatible data:")
    print(f"  Prices shape: {prices.shape}")
    print(f"  Holdings shape: {holdings.shape}")
    
    return batch_data


if __name__ == "__main__":
    test_weight_generation()