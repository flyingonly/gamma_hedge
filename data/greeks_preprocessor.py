"""
Greeks Preprocessor Module
=========================

Unified Greeks preprocessing for all modes (sparse, dense_interpolated, dense_daily_recalc).
All modes use same data reading logic, differ only in timestamp density selection.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.black_scholes import BlackScholesModel
from common.greeks_config import GreeksPreprocessingConfig
from data.options_loader import OptionsDataLoader

class GreeksPreprocessor:
    """Unified Greeks preprocessor for all preprocessing modes"""
    
    def __init__(self, config: Optional[GreeksPreprocessingConfig] = None):
        self.config = config or GreeksPreprocessingConfig()
        self.bs_model = BlackScholesModel()
        self.logger = logging.getLogger(__name__)
        self.data_loader = OptionsDataLoader(
            underlying_data_dir=self.config.underlying_data_dir,
            options_data_dir=self.config.options_data_dir,
            mapping_file=self.config.mapping_file
        )
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.cache_file = os.path.join(self.config.output_dir, 'cache_metadata.json')
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_option_file_path(self, weekly_code: str, option_type: str, strike: float) -> str:
        """Get output file path for preprocessed Greeks with mode-specific subdirectory"""
        filename = f"{option_type}_{strike}_greeks.npz"
        mode_dir = os.path.join(self.config.output_dir, self.config.preprocessing_mode)
        return os.path.join(mode_dir, weekly_code, filename)
    
    def _get_raw_option_file_path(self, weekly_code: str, option_type: str, strike: float) -> str:
        """Get input file path for raw option data"""
        filename = f"{option_type}_{strike}.npz"
        return os.path.join(self.config.options_data_dir, weekly_code, filename)
    
    def _needs_preprocessing(self, weekly_code: str, option_type: str, strike: float) -> bool:
        """Check if option needs preprocessing based on cache"""
        cache_key = f"{weekly_code}_{option_type}_{strike}"
        return cache_key not in self.metadata
    
    def discover_weekly_codes(self) -> List[str]:
        """Discover available weekly codes"""
        codes = []
        if os.path.exists(self.config.options_data_dir):
            for item in os.listdir(self.config.options_data_dir):
                if os.path.isdir(os.path.join(self.config.options_data_dir, item)):
                    codes.append(item)
        return sorted(codes)
    
    def discover_options_for_weekly(self, weekly_code: str) -> List[Tuple[str, float]]:
        """Discover available options for a weekly code"""
        options = []
        weekly_dir = os.path.join(self.config.options_data_dir, weekly_code)
        if os.path.exists(weekly_dir):
            for filename in os.listdir(weekly_dir):
                if filename.endswith('.npz'):
                    parts = filename[:-4].split('_')  # Remove .npz extension
                    if len(parts) >= 2:
                        option_type = parts[0]
                        try:
                            strike = float(parts[1])
                            options.append((option_type, strike))
                        except ValueError:
                            continue
        return sorted(options)
    
    def preprocess_single_option(self, weekly_code: str, option_type: str, strike: float, force_reprocess: bool = False) -> str:
        """Preprocess Greeks for a single option using unified processing"""
        if not force_reprocess and not self._needs_preprocessing(weekly_code, option_type, strike):
            return self._get_option_file_path(weekly_code, option_type, strike)
        
        print(f"Preprocessing {weekly_code} {option_type} {strike}...")
        
        try:
            # Load raw option data
            raw_file = self._get_raw_option_file_path(weekly_code, option_type, strike)
            if not os.path.exists(raw_file):
                raise FileNotFoundError(f"Raw option file not found: {raw_file}")
            
            raw_data = np.load(raw_file, allow_pickle=True)
            option_data = raw_data['data']
            
            # Load underlying data
            underlying_code = self._get_underlying_code(weekly_code)
            underlying_data = self._load_underlying_data(underlying_code)
            
            if underlying_data is None:
                raise ValueError(f"Could not load underlying data for {underlying_code}")
            
            # Unified Greeks calculation
            greeks_data = self._calculate_greeks_unified(
                option_data, underlying_data, option_type, strike, weekly_code
            )
            
            # Save results
            output_file = self._get_option_file_path(weekly_code, option_type, strike)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.savez_compressed(output_file, **greeks_data)
            
            # Update metadata
            cache_key = f"{weekly_code}_{option_type}_{strike}"
            self.metadata[cache_key] = {
                'weekly_code': weekly_code,
                'option_type': option_type,
                'strike': strike,
                'output_file': output_file
            }
            self._save_metadata()
            
            print(f"  Generated: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"  Error preprocessing {weekly_code} {option_type} {strike}: {e}")
            raise
    
    def _calculate_greeks_unified(self, option_data: np.ndarray, underlying_data: Dict, 
                                option_type: str, strike: float, weekly_code: str) -> Dict:
        """
        Unified Greeks calculation for all preprocessing modes.
        All modes read the same data, differ only in time density selection.
        """
        print(f"  Using unified processing with mode: {self.config.preprocessing_mode}")
        
        # Step 1: Standardize all input data to unified format
        standardized_data = self._standardize_input_data(option_data, underlying_data)
        
        # Step 2: Select timestamps based on mode (sparse/dense)
        timestamps = self._select_timestamps_by_mode(standardized_data)
        
        # Step 3: Calculate Greeks for selected timestamps
        return self._calculate_greeks_for_timestamps(
            timestamps, standardized_data, option_type, strike
        )
    
    def _standardize_input_data(self, option_data: np.ndarray, underlying_data: Dict) -> Dict:
        """Convert all data to unified Unix timestamp format"""
        # Process option data
        option_timestamps, option_prices = [], []
        for row in option_data:
            if len(row) >= 2:
                ts = self._convert_timestamp_to_unix(row[0])
                if self._is_valid_timestamp(ts):
                    option_timestamps.append(ts)
                    option_prices.append(float(row[1]))
        
        # Process underlying data  
        underlying_timestamps, underlying_prices = [], []
        for i, timestamp in enumerate(underlying_data['timestamps']):
            ts = self._convert_timestamp_to_unix(timestamp)
            if self._is_valid_timestamp(ts):
                underlying_timestamps.append(ts)
                underlying_prices.append(float(underlying_data['prices'][i]))
        
        return {
            'option_timestamps': np.array(option_timestamps),
            'option_prices': np.array(option_prices),
            'underlying_timestamps': np.array(underlying_timestamps),
            'underlying_prices': np.array(underlying_prices)
        }
    
    def _select_timestamps_by_mode(self, data: Dict) -> np.ndarray:
        """Select timestamps based on preprocessing mode"""
        if self.config.preprocessing_mode == 'sparse':
            return data['option_timestamps']  # Lower density
        else:  # dense_interpolated or dense_daily_recalc
            return data['underlying_timestamps']  # Higher density
    
    def _calculate_greeks_for_timestamps(self, timestamps: np.ndarray, data: Dict,
                                       option_type: str, strike: float) -> Dict:
        """Calculate Greeks for specified timestamps"""
        n_points = len(timestamps)
        result = {
            'timestamps': np.zeros(n_points),
            'underlying_prices': np.zeros(n_points), 
            'option_prices': np.zeros(n_points),
            'delta': np.zeros(n_points),
            'gamma': np.zeros(n_points),
            'theta': np.zeros(n_points),
            'vega': np.zeros(n_points),
            'implied_volatility': np.zeros(n_points)
        }
        
        risk_free_rate = self.config.risk_free_rate
        time_to_expiry = self.config.time_to_expiry_days / 365.0
        valid_count = 0
        
        for i, timestamp in enumerate(timestamps):
            try:
                # Get underlying price
                underlying_price = self._get_price_at_timestamp(
                    timestamp, data['underlying_timestamps'], data['underlying_prices']
                )
                if underlying_price is None:
                    continue
                
                # Get/estimate option price
                option_price = self._get_price_at_timestamp(
                    timestamp, data['option_timestamps'], data['option_prices']
                )
                if option_price is None:
                    # Estimate using Black-Scholes
                    volatility = self._get_volatility_estimate(data, timestamp)
                    option_price = self.bs_model.option_price(
                        underlying_price, strike, time_to_expiry, 
                        risk_free_rate, volatility, option_type.lower()
                    )
                
                # Calculate volatility
                volatility = self._get_volatility_estimate(data, timestamp)
                volatility = np.clip(volatility, self.config.min_volatility, self.config.max_volatility)
                
                # Calculate all Greeks
                greeks = self.bs_model.calculate_all(
                    underlying_price, strike, time_to_expiry, 
                    risk_free_rate, volatility, option_type.lower()
                )
                
                # Store results
                result['timestamps'][valid_count] = timestamp
                result['underlying_prices'][valid_count] = underlying_price
                result['option_prices'][valid_count] = option_price
                result['delta'][valid_count] = greeks['delta']
                result['gamma'][valid_count] = greeks['gamma']
                result['theta'][valid_count] = greeks['theta']
                result['vega'][valid_count] = greeks['vega']
                result['implied_volatility'][valid_count] = volatility
                
                valid_count += 1
                
            except Exception as e:
                continue
        
        print(f"  Success rate: {valid_count}/{n_points} ({valid_count/n_points*100:.1f}%)")
        
        # Trim arrays
        for key in result:
            result[key] = result[key][:valid_count]
            
        return result
    
    def _get_price_at_timestamp(self, target_ts: float, timestamps: np.ndarray, prices: np.ndarray) -> Optional[float]:
        """Get price at timestamp with tolerance"""
        if len(timestamps) == 0:
            return None
        
        time_diffs = np.abs(timestamps - target_ts)
        closest_idx = np.argmin(time_diffs)
        tolerance = self.config.time_matching_tolerance_hours * 3600
        
        if time_diffs[closest_idx] < tolerance:
            return float(prices[closest_idx])
        return None
    
    def _get_volatility_estimate(self, data: Dict, timestamp: float) -> float:
        """Estimate volatility using historical data"""
        try:
            return self._calculate_historical_volatility(
                data['underlying_timestamps'], data['underlying_prices'], timestamp
            )
        except:
            return self.config.default_volatility
    
    def _calculate_historical_volatility(self, timestamps: np.ndarray, prices: np.ndarray, target_timestamp: float) -> float:
        """Calculate historical volatility using recent price data"""
        try:
            # Use 30-day window for volatility calculation  
            window_seconds = 30 * 24 * 3600
            start_time = target_timestamp - window_seconds
            
            # Filter prices within window
            mask = (timestamps >= start_time) & (timestamps <= target_timestamp)
            window_prices = prices[mask]
            
            if len(window_prices) < 2:
                return self.config.default_volatility
            
            # Calculate log returns
            returns = np.diff(np.log(window_prices))
            
            # Annualized volatility
            volatility = np.std(returns) * np.sqrt(252)  # 252 trading days
            
            return volatility if volatility > 0 else self.config.default_volatility
            
        except Exception:
            return self.config.default_volatility
    
    def _is_valid_timestamp(self, timestamp: float) -> bool:
        """Check timestamp validity (2000-2035)"""
        min_ts = datetime(2000, 1, 1).timestamp()
        max_ts = datetime(2035, 12, 31).timestamp()
        return min_ts <= timestamp <= max_ts and timestamp > 0
    
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
            elif isinstance(timestamp, (int, float, np.integer, np.floating)):
                return float(timestamp)
            # Handle string datetime
            else:
                return pd.to_datetime(timestamp).timestamp()
        except Exception as e:
            return 0.0
    
    def _get_underlying_code(self, weekly_code: str) -> str:
        """Map weekly code to underlying code"""
        if '3C' in weekly_code:
            return 'USU5'
        elif '3I' in weekly_code:
            return 'FVU5'
        elif '3M' in weekly_code:
            return 'TUU5'
        elif '3W' in weekly_code:
            return 'TYU5'
        else:
            return 'USU5'
    
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
            print(f"Error loading underlying data {underlying_code}: {e}")
            return None