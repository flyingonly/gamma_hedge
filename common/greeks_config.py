"""
Unified Data Processing Configuration
====================================

Single, simplified configuration for all data processing operations.
Replaces all previous distributed configuration classes.
"""

from dataclasses import dataclass


@dataclass
class UnifiedDataConfig:
    """
    Unified configuration for all data processing operations.
    Single source of truth for paths, parameters, and settings.
    """
    # Data paths
    underlying_data_dir: str = "csv_process/underlying_npz"
    options_data_dir: str = "csv_process/weekly_options_data"
    mapping_file: str = "csv_process/weekly_options_data/weekly_mapping.json"
    preprocessed_greeks_dir: str = "data/preprocessed_greeks"
    
    # Black-Scholes parameters
    risk_free_rate: float = 0.05
    time_to_expiry_days: float = 7.0
    min_volatility: float = 0.01
    max_volatility: float = 2.0
    default_volatility: float = 0.20
    volatility_window_days: int = 30
    
    # Processing parameters
    batch_size: int = 1000
    parallel_processing: bool = True
    time_matching_tolerance_hours: int = 24  # Relaxed from 1 hour to 24 hours
    
    # Output settings
    compression: bool = True
    verbose: bool = True


# Primary configuration class used by all modules
GreeksPreprocessingConfig = UnifiedDataConfig