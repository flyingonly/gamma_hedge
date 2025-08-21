"""
Greeks Preprocessor Module
=========================

Preprocesses Greeks (Delta, Gamma, Theta, Vega) for individual options.
Reads raw option data and calculates Greeks using Black-Scholes model.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.black_scholes import BlackScholesModel
from common.greeks_config import GreeksPreprocessingConfig
from data.options_loader import OptionsDataLoader

class GreeksPreprocessor:
    """
    Preprocesses Greeks data for individual options.
    
    Reads raw option data from csv_process directory and calculates
    Greeks values using Black-Scholes model for each timestamp.
    """
    
    def __init__(self, config: Optional[GreeksPreprocessingConfig] = None):
        """
        Initialize Greeks preprocessor.
        
        Args:
            config: Delta hedge configuration, creates default if None
        """
        self.config = config or GreeksPreprocessingConfig()
        self.bs_model = BlackScholesModel()
        # Use new unified data loader
        self.data_loader = OptionsDataLoader(
            underlying_data_dir=self.config.underlying_data_dir,
            options_data_dir=self.config.options_data_dir,
            mapping_file=self.config.mapping_file
        )
        
        # Create output directory
        self.output_dir = os.path.join('data', 'preprocessed_greeks')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = os.path.join(self.output_dir, 'cache_metadata.json')
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_option_file_path(self, weekly_code: str, option_type: str, strike: float) -> str:
        """Get path for preprocessed option Greeks file"""
        option_dir = os.path.join(self.output_dir, weekly_code)
        os.makedirs(option_dir, exist_ok=True)
        filename = f"{option_type}_{strike}_greeks.npz"
        return os.path.join(option_dir, filename)
    
    def _get_raw_option_file_path(self, weekly_code: str, option_type: str, strike: float) -> str:
        """Get path for raw option data file from csv_process"""
        filename = f"{option_type}_{strike}.npz"
        return os.path.join('csv_process', 'weekly_options_data', weekly_code, filename)
    
    def _needs_preprocessing(self, weekly_code: str, option_type: str, strike: float) -> bool:
        """Check if option needs preprocessing (file missing or outdated)"""
        greeks_file = self._get_option_file_path(weekly_code, option_type, strike)
        raw_file = self._get_raw_option_file_path(weekly_code, option_type, strike)
        
        # Check if files exist
        if not os.path.exists(greeks_file) or not os.path.exists(raw_file):
            return True
        
        # Check modification times
        option_key = f"{weekly_code}/{option_type}_{strike}"
        if option_key in self.metadata:
            raw_mtime = os.path.getmtime(raw_file)
            cached_mtime = self.metadata[option_key].get('raw_file_mtime', 0)
            return raw_mtime > cached_mtime
        
        return True
    
    def preprocess_single_option(self, weekly_code: str, option_type: str, strike: float, force_reprocess: bool = False) -> str:
        """
        Preprocess Greeks for a single option.
        
        Args:
            weekly_code: Weekly code like '3CN5'
            option_type: 'CALL' or 'PUT'
            strike: Strike price
            force_reprocess: If True, bypass cache and reprocess file
            
        Returns:
            Path to generated Greeks file
        """
        if not force_reprocess and not self._needs_preprocessing(weekly_code, option_type, strike):
            return self._get_option_file_path(weekly_code, option_type, strike)
        
        print(f"Preprocessing {weekly_code} {option_type} {strike}...")
        
        try:
            # Load raw option data
            raw_file = self._get_raw_option_file_path(weekly_code, option_type, strike)
            if not os.path.exists(raw_file):
                raise FileNotFoundError(f"Raw option file not found: {raw_file}")
            
            raw_data = np.load(raw_file, allow_pickle=True)
            option_data = raw_data['data']  # Time series data
            
            # Get underlying data for this weekly code
            underlying_code = self._get_underlying_code(weekly_code)
            underlying_data = self._load_underlying_data(underlying_code)
            
            if underlying_data is None:
                raise ValueError(f"Could not load underlying data for {underlying_code}")
            
            # Calculate Greeks for each timestamp
            greeks_data = self._calculate_greeks_timeseries(
                option_data, underlying_data, option_type, strike, weekly_code
            )
            
            # Save preprocessed Greeks
            output_file = self._get_option_file_path(weekly_code, option_type, strike)
            np.savez_compressed(output_file, **greeks_data)
            
            # Update metadata
            option_key = f"{weekly_code}/{option_type}_{strike}"
            self.metadata[option_key] = {
                'processed_time': datetime.now().isoformat(),
                'raw_file_mtime': os.path.getmtime(raw_file),
                'output_file': output_file
            }
            self._save_metadata()
            
            print(f"  Generated: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"  Error preprocessing {weekly_code} {option_type} {strike}: {e}")
            raise
    
    def _get_underlying_code(self, weekly_code: str) -> str:
        """Map weekly code to underlying code"""
        # Simple mapping - can be made more sophisticated
        if '3C' in weekly_code:
            return 'USU5'  # Corn -> 10-year treasury note
        elif '3I' in weekly_code:
            return 'FVU5'  # Iron ore -> 5-year treasury note  
        elif '3M' in weekly_code:
            return 'TUU5'  # Meal -> 2-year treasury note
        elif '3W' in weekly_code:
            return 'TYU5'  # Wheat -> 10-year treasury note
        else:
            return 'USU5'  # Default
    
    def _load_underlying_data(self, underlying_code: str) -> Optional[Dict]:
        """Load underlying asset data"""
        try:
            underlying_file = os.path.join('csv_process', 'underlying_npz', f'{underlying_code}.npz')
            if not os.path.exists(underlying_file):
                print(f"Warning: Underlying file not found: {underlying_file}")
                return None
                
            data = np.load(underlying_file, allow_pickle=True)
            return {
                'timestamps': data['timestamps'],
                'prices': data['prices']
            }
        except Exception as e:
            print(f"Error loading underlying data for {underlying_code}: {e}")
            return None
    
    def _calculate_greeks_timeseries(self, option_data: np.ndarray, underlying_data: Dict, 
                                   option_type: str, strike: float, weekly_code: str) -> Dict:
        """Calculate Greeks for entire time series with fixed timestamp handling"""
        
        n_points = len(option_data)
        
        # Initialize arrays
        greeks_data = {
            'timestamps': np.zeros(n_points),
            'underlying_prices': np.zeros(n_points), 
            'option_prices': np.zeros(n_points),
            'delta': np.zeros(n_points),
            'gamma': np.zeros(n_points),
            'theta': np.zeros(n_points),
            'vega': np.zeros(n_points),
            'implied_volatility': np.zeros(n_points)
        }
        
        # Use config parameters
        risk_free_rate = self.config.risk_free_rate
        time_to_expiry = self.config.time_to_expiry_days / 365.0
        
        valid_count = 0
        
        for i in range(n_points):
            try:
                # Extract data for this timestamp
                if len(option_data[i]) >= 2:
                    timestamp = option_data[i][0]
                    option_price = option_data[i][1]
                else:
                    continue
                
                # Fix timestamp conversion - handle pandas.Timestamp properly
                timestamp_unix = self._convert_timestamp_to_unix(timestamp)
                
                # Find corresponding underlying price with relaxed tolerance
                underlying_price = self._get_underlying_price_at_time(
                    underlying_data, timestamp_unix
                )
                
                if underlying_price is None:
                    continue
                
                # Calculate volatility
                volatility = self._get_volatility_estimate(underlying_data, timestamp_unix)
                volatility = max(self.config.min_volatility, 
                               min(self.config.max_volatility, volatility))
                
                # Calculate Greeks using Black-Scholes
                delta = self.bs_model.delta(
                    underlying_price, strike, time_to_expiry, 
                    risk_free_rate, volatility, option_type.lower()
                )
                
                gamma = self.bs_model.gamma(
                    underlying_price, strike, time_to_expiry,
                    risk_free_rate, volatility
                )
                
                theta = self.bs_model.theta(
                    underlying_price, strike, time_to_expiry,
                    risk_free_rate, volatility, option_type.lower()
                )
                
                vega = self.bs_model.vega(
                    underlying_price, strike, time_to_expiry,
                    risk_free_rate, volatility
                )
                
                # Store results using valid_count index to compact array
                greeks_data['timestamps'][valid_count] = timestamp_unix
                greeks_data['underlying_prices'][valid_count] = underlying_price
                greeks_data['option_prices'][valid_count] = option_price
                greeks_data['delta'][valid_count] = delta
                greeks_data['gamma'][valid_count] = gamma  
                greeks_data['theta'][valid_count] = theta
                greeks_data['vega'][valid_count] = vega
                greeks_data['implied_volatility'][valid_count] = volatility
                
                valid_count += 1
                
            except Exception as e:
                print(f"  Warning: Failed to calculate Greeks for point {i}: {e}")
                continue
        
        print(f"  Input: {n_points} data points → Output: {valid_count} data points ({valid_count/n_points*100:.1f}% success rate)")
        
        # Trim arrays to actual valid data count
        for key in greeks_data:
            greeks_data[key] = greeks_data[key][:valid_count]
            
        return greeks_data
    
    def _convert_timestamp_to_unix(self, timestamp) -> float:
        """Convert various timestamp formats to Unix timestamp"""
        try:
            # Handle pandas.Timestamp
            if hasattr(timestamp, 'timestamp'):
                return timestamp.timestamp()
            # Handle numpy datetime64
            elif hasattr(timestamp, 'astype'):
                return timestamp.astype('datetime64[s]').astype(float)
            # Handle float/int (already Unix timestamp)
            elif isinstance(timestamp, (int, float)):
                return float(timestamp)
            # Handle string datetime
            else:
                return pd.to_datetime(timestamp).timestamp()
        except Exception as e:
            print(f"  Warning: Failed to convert timestamp {timestamp}: {e}")
            return 0.0
    
    def _get_underlying_price_at_time(self, underlying_data: Dict, timestamp: float) -> Optional[float]:
        """Get underlying price at specific timestamp with relaxed tolerance"""
        try:
            timestamps = underlying_data['timestamps']
            prices = underlying_data['prices']
            
            # Find closest timestamp
            time_diffs = np.abs(timestamps - timestamp)
            closest_idx = np.argmin(time_diffs)
            
            # Use relaxed time window (24 hours = 86400 seconds instead of 1 hour)
            tolerance_seconds = self.config.time_matching_tolerance_hours * 3600
            if time_diffs[closest_idx] < tolerance_seconds:
                return float(prices[closest_idx])
            
            return None
            
        except Exception as e:
            return None
    
    def _get_volatility_estimate(self, underlying_data: Dict, timestamp: float) -> float:
        """
        Estimate volatility using historical data or configured default.
        
        Args:
            underlying_data: Underlying asset data with prices and timestamps
            timestamp: Current timestamp
            
        Returns:
            Estimated volatility (annualized)
        """
        try:
            timestamps = underlying_data['timestamps']
            prices = underlying_data['prices']
            
            # Find data points within volatility window
            window_seconds = self.config.volatility_window_days * 24 * 3600  # Convert days to seconds
            mask = (timestamps >= timestamp - window_seconds) & (timestamps <= timestamp)
            
            if mask.sum() < 5:  # Need minimum data points
                return self.config.default_volatility
            
            # Calculate historical volatility
            window_prices = prices[mask]
            returns = np.diff(np.log(window_prices))
            
            if len(returns) < 2:
                return self.config.default_volatility
            
            # Annualize volatility (assuming intraday data)
            # Number of observations per year depends on data frequency
            observations_per_day = len(returns) / (mask.sum() / (24 * 3600))  # Rough estimate
            annualization_factor = np.sqrt(252 * observations_per_day)  # 252 trading days
            
            historical_vol = np.std(returns) * annualization_factor
            return float(historical_vol)
            
        except Exception:
            # Any calculation error falls back to configured default
            return self.config.default_volatility
    
    def discover_weekly_codes(self, base_path: str = 'csv_process/weekly_options_data') -> List[str]:
        """Discover all available weekly codes"""
        if not os.path.exists(base_path):
            return []
        
        return [d for d in os.listdir(base_path) 
                if os.path.isdir(os.path.join(base_path, d))]
    
    def discover_options_for_weekly(self, weekly_code: str) -> List[Tuple[str, float]]:
        """Discover all options for a weekly code"""
        weekly_path = os.path.join('csv_process', 'weekly_options_data', weekly_code)
        if not os.path.exists(weekly_path):
            return []
        
        options = []
        for filename in os.listdir(weekly_path):
            if filename.endswith('.npz') and '_' in filename:
                try:
                    parts = filename.replace('.npz', '').split('_')
                    if len(parts) == 2:
                        option_type = parts[0]
                        strike = float(parts[1])
                        options.append((option_type, strike))
                except ValueError:
                    continue
        
        return options
    
    def batch_preprocess(self, weekly_codes: List[str], force_reprocess: bool = False) -> Dict[str, List[str]]:
        """
        Batch preprocess all options for given weekly codes.
        
        Args:
            weekly_codes: List of weekly codes to process
            force_reprocess: If True, bypass cache and reprocess all files
            
        Returns:
            Dict mapping weekly_code -> list of generated file paths
        """
        results = {}
        
        for weekly_code in weekly_codes:
            print(f"Processing weekly code: {weekly_code}")
            results[weekly_code] = []
            
            options = self.discover_options_for_weekly(weekly_code)
            print(f"  Found {len(options)} options")
            
            for option_type, strike in options:
                try:
                    output_file = self.preprocess_single_option(weekly_code, option_type, strike, force_reprocess=force_reprocess)
                    results[weekly_code].append(output_file)
                except Exception as e:
                    print(f"  Failed to process {option_type} {strike}: {e}")
                    continue
        
        return results