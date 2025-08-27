"""
Options Data Loader Module
==========================

Unified options data loader for underlying assets and options data.
Integrates data from underlying_npz and weekly_options_data directories.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from utils.logger import get_logger
import logging
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csv_process.separated_npz_loader import SeparatedNPZLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class OptionsDataLoader:
    """
    Unified data loader for underlying assets and options data.
    
    Provides interface to load, align, and preprocess:
    - Underlying asset price data from NPZ files
    - Options contract data from weekly_options_data
    - Contract specifications from mapping files
    """
    
    def __init__(self, underlying_data_dir: str = "csv_process/underlying_npz",
                 options_data_dir: str = "csv_process/weekly_options_data",
                 mapping_file: str = "csv_process/monthly_mapping.json"):
        """
        Initialize data loader with data paths.
        
        Args:
            underlying_data_dir: Path to underlying NPZ data
            options_data_dir: Path to weekly options data
            mapping_file: Path to options mapping JSON file
        """
        self.underlying_data_dir = underlying_data_dir
        self.options_data_dir = options_data_dir
        self.mapping_file = mapping_file
        
        self.underlying_loader = None
        self.options_mapping = None
        self.loaded_underlying_data = {}
        self.loaded_options_data = {}
        
        # Initialize components
        self._initialize_underlying_loader()
        self._load_options_mapping()
        
        logger.info("OptionsDataLoader initialized successfully")
    
    def _initialize_underlying_loader(self):
        """Initialize the underlying data loader"""
        try:
            self.underlying_loader = SeparatedNPZLoader(
                data_dir=self.underlying_data_dir
            )
            logger.info(f"Loaded underlying data loader with {len(self.underlying_loader.get_codes())} assets")
        except Exception as e:
            logger.error(f"Failed to initialize underlying loader: {e}")
            raise
    
    def _load_options_mapping(self):
        """Load options mapping information from JSON file"""
        try:
            mapping_path = Path(self.mapping_file)
            if not mapping_path.exists():
                raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
            
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.options_mapping = json.load(f)
            
            logger.info(f"Loaded options mapping with {len(self.options_mapping)} file entries")
        except Exception as e:
            logger.error(f"Failed to load options mapping: {e}")
            raise
    
    def get_available_underlyings(self) -> List[str]:
        """Get list of available underlying asset codes"""
        if self.underlying_loader:
            return self.underlying_loader.get_codes()
        return []
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """
        Get available options contracts grouped by underlying.
        
        Returns:
            Dictionary mapping underlying codes to list of weekly option codes
        """
        options_by_underlying = {}
        
        if not self.options_mapping:
            return options_by_underlying
        
        for file_entry in self.options_mapping.values():
            for product in file_entry.get('products', []):
                underlying = product.get('underlying')
                weekly_code = product.get('weekly_code')
                
                if underlying and weekly_code:
                    if underlying not in options_by_underlying:
                        options_by_underlying[underlying] = []
                    if weekly_code not in options_by_underlying[underlying]:
                        options_by_underlying[underlying].append(weekly_code)
        
        return options_by_underlying
    
    def load_underlying_data(self, underlying_code: str, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Dict[str, np.ndarray]:
        """
        Load underlying asset price data.
        
        Args:
            underlying_code: Asset code (e.g., 'USU5', 'FVU5')
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary with keys: 'timestamps', 'prices', 'volumes'
        """
        try:
            if underlying_code in self.loaded_underlying_data:
                data = self.loaded_underlying_data[underlying_code]
            else:
                timestamps, prices, volumes = self.underlying_loader.load_arrays(underlying_code)
                data = {
                    'timestamps': timestamps,
                    'prices': prices,
                    'volumes': volumes
                }
                self.loaded_underlying_data[underlying_code] = data
            
            # Apply date filters if specified
            if start_date or end_date:
                data = self._filter_by_date(data, start_date, end_date)
            
            logger.info(f"Loaded {len(data['prices'])} data points for {underlying_code}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load underlying data for {underlying_code}: {e}")
            raise
    
    def load_options_data(self, weekly_code: str, 
                         option_type: Optional[str] = None,
                         strike_range: Optional[Tuple[float, float]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load options data for a specific weekly contract.
        
        Args:
            weekly_code: Weekly option code (e.g., '3CN5', '3IN5')
            option_type: Filter by 'CALL' or 'PUT', None for both
            strike_range: Optional (min_strike, max_strike) filter
            
        Returns:
            Dictionary mapping option identifiers to DataFrames with columns:
            ['datetime', 'last_price', 'volume', 'strike', 'weekly_code', 'option_type']
        """
        cache_key = f"{weekly_code}_{option_type}_{strike_range}"
        if cache_key in self.loaded_options_data:
            return self.loaded_options_data[cache_key]
        
        options_data = {}
        options_dir = Path(self.options_data_dir) / weekly_code
        
        if not options_dir.exists():
            logger.warning(f"Options directory not found: {options_dir}")
            return options_data
        
        # Determine which option types to load
        types_to_load = ['CALL', 'PUT'] if option_type is None else [option_type]
        
        for opt_type in types_to_load:
            pattern = f"{opt_type}_*.npz"
            option_files = list(options_dir.glob(pattern))
            
            for file_path in option_files:
                try:
                    # Extract strike from filename
                    strike_str = file_path.stem.replace(f"{opt_type}_", "")
                    strike = float(strike_str)
                    
                    # Apply strike filter if specified
                    if strike_range and not (strike_range[0] <= strike <= strike_range[1]):
                        continue
                    
                    # Load NPZ data
                    data = np.load(file_path, allow_pickle=True)
                    
                    # Create DataFrame
                    df_data = []
                    for i in range(len(data['data'])):
                        row = data['data'][i]
                        df_data.append({
                            'datetime': row[0],
                            'last_price': row[1],
                            'volume': row[2],
                            'strike': row[3],
                            'weekly_code': row[4],
                            'option_type': row[5]
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        # Convert datetime if needed
                        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                            df['datetime'] = pd.to_datetime(df['datetime'])
                        
                        option_id = f"{opt_type}_{strike}"
                        options_data[option_id] = df
                        
                except Exception as e:
                    logger.warning(f"Failed to load option file {file_path}: {e}")
                    continue
        
        self.loaded_options_data[cache_key] = options_data
        logger.info(f"Loaded {len(options_data)} option contracts for {weekly_code}")
        return options_data
    
    def get_option_contract_info(self, weekly_code: str) -> Optional[Dict]:
        """
        Get contract specifications for a weekly option.
        
        Args:
            weekly_code: Weekly option code
            
        Returns:
            Dictionary with contract information including expiry_date, underlying, etc.
        """
        if not self.options_mapping:
            return None
        
        for file_entry in self.options_mapping.values():
            for product in file_entry.get('products', []):
                if product.get('weekly_code') == weekly_code:
                    return product
        
        return None
    
    def _filter_by_date(self, data: Dict[str, np.ndarray],
                       start_date: Optional[datetime],
                       end_date: Optional[datetime]) -> Dict[str, np.ndarray]:
        """Filter data by date range"""
        timestamps = pd.to_datetime(data['timestamps'], unit='s')
        
        mask = pd.Series(True, index=timestamps.index)
        if start_date:
            mask &= (timestamps >= start_date)
        if end_date:
            mask &= (timestamps <= end_date)
        
        return {
            'timestamps': data['timestamps'][mask],
            'prices': data['prices'][mask],
            'volumes': data['volumes'][mask]
        }
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of available data"""
        summary = {
            'underlyings': {
                'available_codes': self.get_available_underlyings(),
                'count': len(self.get_available_underlyings())
            },
            'options': {
                'available_by_underlying': self.get_available_options(),
                'total_weekly_codes': sum(len(codes) for codes in self.get_available_options().values())
            },
            'loaded_data': {
                'underlying_cache_size': len(self.loaded_underlying_data),
                'options_cache_size': len(self.loaded_options_data)
            }
        }
        
        return summary


# Utility functions for data validation and preprocessing

def validate_data_quality(data: Dict[str, np.ndarray], 
                         missing_threshold: float = 0.1) -> Dict[str, bool]:
    """
    Validate data quality and completeness.
    
    Args:
        data: Data dictionary to validate
        missing_threshold: Maximum allowed ratio of missing data
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    for key, values in data.items():
        if isinstance(values, np.ndarray):
            # Check for missing values
            missing_ratio = np.isnan(values).sum() / len(values)
            validation_results[f"{key}_missing_ok"] = missing_ratio <= missing_threshold
            
            # Check for infinite values
            validation_results[f"{key}_finite"] = np.isfinite(values).all()
            
            # Check for reasonable ranges (basic sanity check)
            if key == 'prices':
                validation_results[f"{key}_positive"] = (values > 0).all()
    
    return validation_results


def calculate_data_statistics(data: Dict[str, np.ndarray]) -> Dict:
    """Calculate basic statistics for data arrays"""
    stats = {}
    
    for key, values in data.items():
        if isinstance(values, np.ndarray) and values.dtype in [np.float32, np.float64]:
            stats[key] = {
                'mean': float(np.nanmean(values)),
                'std': float(np.nanstd(values)),
                'min': float(np.nanmin(values)),
                'max': float(np.nanmax(values)),
                'count': len(values),
                'missing_count': int(np.isnan(values).sum())
            }
    
    return stats