"""
Unified Configuration System for Gamma Hedge

This module replaces the fragmented configuration system with a single, unified
configuration management approach following SOLID principles.

Replaces:
- common/config_manager.py
- common/enhanced_config_manager.py  
- common/config_sources.py
- common/config_validator.py
- utils/config.py (partially)

Priority Order: CLI Arguments > Environment Variables > Config Files > Defaults
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, fields
from copy import deepcopy
logger = logging.getLogger(__name__)


@dataclass
class MarketConfig:
    """Market simulation parameters"""
    n_assets: int = 3
    n_timesteps: int = 50
    initial_price: float = 100.0
    volatility: float = 0.2
    dt: float = 0.01
    drift: float = 0.05
    initial_holding: List[float] = None
    target_holding: List[float] = None
    seed: Optional[int] = 42
    
    def __post_init__(self):
        if self.initial_holding is None:
            self.initial_holding = [0.0] * self.n_assets
        if self.target_holding is None:
            self.target_holding = [1.0] * self.n_assets


@dataclass 
class TrainingConfig:
    """Training and optimization parameters"""
    batch_size: int = 32
    learning_rate: float = 0.001
    n_epochs: int = 10
    n_train_samples: int = 10000
    n_val_samples: int = 2000
    device: str = "cuda"
    checkpoint_interval: int = 10
    early_stopping_patience: Optional[int] = None
    
    # Regularization parameters (unified from multiple sources)
    entropy_weight: float = 0.01
    base_execution_cost: float = 0.001
    variable_execution_cost: float = 0.0001
    market_impact_scale: float = 0.01
    
    # Loss component weights (previously hardcoded)
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    l2_regularization_weight: float = 0.0001
    
    # Execution probability variation enhancement parameters
    sharpness_regularizer: float = 0.1
    entropy_gate_k: float = 3.0
    
    # Temporal Edge-Aware Regularizer (TEAR) parameters
    temporal_push: float = 0.05
    temporal_smooth: float = 0.01
    
    # Time series parameters
    align_to_daily: bool = False
    split_ratios: List[float] = None
    min_daily_sequences: int = 50
    
    def __post_init__(self):
        if self.split_ratios is None:
            self.split_ratios = [0.8, 0.2, 0.0]  # train, val, test
        
        # Validate split ratios
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(self.split_ratios)}")


@dataclass
class ModelConfig:
    """Neural network model parameters"""
    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    input_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


@dataclass
class DeltaHedgeConfig:
    """Delta hedging specific parameters"""
    enable_delta: bool = False
    underlying_codes: List[str] = None
    weekly_codes: List[str] = None
    sequence_length: int = 100
    underlying_dense_mode: bool = False
    
    def __post_init__(self):
        if self.underlying_codes is None:
            self.underlying_codes = ["USU5", "FVU5"]
        if self.weekly_codes is None:
            self.weekly_codes = ["3CN5"]


@dataclass
class LoggingConfig:
    """Unified logging system parameters"""
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: Optional[str] = None  # Optional log file path
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_output: bool = True  # Enable console output
    file_output: bool = False  # Enable file output (requires log_file)
    
    def __post_init__(self):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        self.log_level = self.log_level.upper()
        
        # Validate file output configuration
        if self.file_output and not self.log_file:
            raise ValueError("log_file must be specified when file_output=True")


@dataclass
class GreeksConfig:
    """Greeks preprocessing parameters"""
    preprocessing_mode: str = "sparse"  # sparse, dense_interpolated, dense_daily_recalc
    force_reprocess: bool = True
    resume_from_cache: bool = False
    risk_free_rate: float = 0.02
    dividend_yield: float = 0.0
    time_to_expiry_days: int = 30
    volatility_fallback: float = 0.2
    
    # Data directories
    underlying_data_dir: str = "csv_process/underlying_npz"
    options_data_dir: str = "csv_process/weekly_options_data"
    output_dir: str = "data/preprocessed_greeks"
    mapping_file: str = "csv_process/weekly_options_data/weekly_mapping.json"
    
    # Processing parameters
    min_volatility: float = 0.01
    max_volatility: float = 3.0
    default_volatility: float = 0.2
    time_matching_tolerance_hours: int = 24
    
    def __post_init__(self):
        valid_modes = ["sparse", "dense_interpolated", "dense_daily_recalc"]
        if self.preprocessing_mode not in valid_modes:
            raise ValueError(f"preprocessing_mode must be one of {valid_modes}")


class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass


class Config:
    """
    Unified configuration management system
    
    Provides a single interface for all configuration needs, replacing the
    fragmented configuration system with clear priority hierarchy.
    
    Priority: CLI > Environment Variables > Config Files > Defaults
    """
    
    def __init__(self, 
                 cli_args: Optional[Dict[str, Any]] = None,
                 config_file: Optional[Union[str, Path]] = None,
                 env_prefix: str = "GAMMA_HEDGE"):
        """
        Initialize unified configuration
        
        Args:
            cli_args: Command line arguments dictionary
            config_file: Path to configuration file (JSON)
            env_prefix: Environment variable prefix
        """
        self.env_prefix = env_prefix
        self._config_data = {}
        self._load_order = []
        
        # Load configuration in priority order
        self._load_defaults()
        self._load_config_file(config_file)
        self._load_environment_variables()
        self._load_cli_arguments(cli_args or {})
        
        # Create typed configuration objects
        self._create_config_objects()
        
        # Validate configuration
        self.validate()
        
        logger.info(f"Configuration loaded successfully. Sources: {', '.join(self._load_order)}")
    
    def _load_defaults(self):
        """Load default configuration values"""
        defaults = {
            'market': asdict(MarketConfig()),
            'training': asdict(TrainingConfig()),
            'model': asdict(ModelConfig()), 
            'delta_hedge': asdict(DeltaHedgeConfig()),
            'greeks': asdict(GreeksConfig()),
            'logging': asdict(LoggingConfig())
        }
        
        self._config_data.update(defaults)
        self._load_order.append("defaults")
    
    def _load_config_file(self, config_file: Optional[Union[str, Path]]):
        """Load configuration from JSON file"""
        if not config_file:
            # Try default config files
            for default_path in ["config/default.json", "config/development.json"]:
                if os.path.exists(default_path):
                    config_file = default_path
                    break
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Deep merge configuration
                self._deep_merge(self._config_data, file_config)
                self._load_order.append(f"file({config_file})")
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_updates = {}
        
        for key, value in os.environ.items():
            if key.startswith(f"{self.env_prefix}_"):
                # Convert environment variable to config path
                config_path = key[len(f"{self.env_prefix}_"):].lower().replace("_", ".")
                self._set_nested_value(env_updates, config_path, self._parse_env_value(value))
        
        if env_updates:
            self._deep_merge(self._config_data, env_updates)
            self._load_order.append("environment")
    
    def _load_cli_arguments(self, cli_args: Dict[str, Any]):
        """Load configuration from command line arguments"""
        if not cli_args:
            return
        
        # Convert CLI arguments to configuration format
        cli_config = {}
        
        # Map common CLI arguments to configuration paths
        cli_mapping = {
            'batch_size': 'training.batch_size',
            'learning_rate': 'training.learning_rate', 
            'epochs': 'training.n_epochs',
            'device': 'training.device',
            'portfolio': 'training.portfolio_name',
            'align_to_daily': 'training.align_to_daily',
            'split_ratios': 'training.split_ratios',
            'entropy_weight': 'training.entropy_weight',
            'policy_loss_weight': 'training.policy_loss_weight',
            'value_loss_weight': 'training.value_loss_weight',
            'l2_regularization_weight': 'training.l2_regularization_weight',
            'preprocessing_mode': 'greeks.preprocessing_mode',
            'underlying_dense_mode': 'delta_hedge.underlying_dense_mode',
            'sequence_length': 'delta_hedge.sequence_length'
        }
        
        for cli_key, config_path in cli_mapping.items():
            if cli_key in cli_args and cli_args[cli_key] is not None:
                self._set_nested_value(cli_config, config_path, cli_args[cli_key])
        
        if cli_config:
            self._deep_merge(self._config_data, cli_config)
            self._load_order.append("CLI")
    
    def _create_config_objects(self):
        """Create typed configuration objects from raw data"""
        try:
            self.market = MarketConfig(**self._config_data['market'])
            self.training = TrainingConfig(**self._config_data['training'])
            self.model = ModelConfig(**self._config_data['model'])
            self.delta_hedge = DeltaHedgeConfig(**self._config_data['delta_hedge'])
            self.greeks = GreeksConfig(**self._config_data['greeks'])
            self.logging = LoggingConfig(**self._config_data['logging'])
        except TypeError as e:
            raise ConfigValidationError(f"Failed to create configuration objects: {e}")
    
    def validate(self):
        """Validate configuration consistency and constraints"""
        errors = []
        
        # Training configuration validation
        if self.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if self.training.n_epochs <= 0:
            errors.append("n_epochs must be positive")
        
        # Market configuration validation
        if self.market.n_assets <= 0:
            errors.append("n_assets must be positive")
        
        if self.market.volatility <= 0:
            errors.append("volatility must be positive")
        
        # Model configuration validation  
        if self.model.input_dim is not None and self.model.input_dim <= 0:
            errors.append("input_dim must be positive if specified")
        
        # Delta hedge configuration validation
        if self.delta_hedge.enable_delta:
            if self.delta_hedge.sequence_length <= 0:
                errors.append("sequence_length must be positive for delta hedging")
        
        # Greeks configuration validation
        if self.greeks.time_to_expiry_days <= 0:
            errors.append("time_to_expiry_days must be positive")
        
        if errors:
            raise ConfigValidationError(f"Configuration validation errors: {'; '.join(errors)}")
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value with dot notation"""
        return self._get_nested_value(self._config_data, key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value with dot notation"""
        self._set_nested_value(self._config_data, key, value)
        # Recreate config objects with new data
        self._create_config_objects()
        self.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return deepcopy(self._config_data)
    
    def save(self, filepath: Union[str, Path]):
        """Save current configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self._config_data, f, indent=2)
    
    @staticmethod
    def _deep_merge(base_dict: Dict, update_dict: Dict):
        """Deep merge dictionaries, updating base_dict in place"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                Config._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    @staticmethod
    def _get_nested_value(data: Dict, key: str, default=None) -> Any:
        """Get nested dictionary value using dot notation"""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    @staticmethod
    def _set_nested_value(data: Dict, key: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key.split('.')
        current = data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable string to appropriate type"""
        # Try to parse as JSON first (for lists, dicts, etc.)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access for backward compatibility"""
        # Map old-style keys to new configuration objects
        if key == 'market':
            return self.market
        elif key == 'training':
            return self.training
        elif key == 'model':
            return self.model
        elif key == 'delta_hedge':
            return self.delta_hedge
        elif key == 'greeks':
            return self.greeks
        else:
            return self.get(key)
    
    def __repr__(self) -> str:
        return f"Config(sources={self._load_order})"


def create_config(cli_args: Optional[Dict[str, Any]] = None,
                 config_file: Optional[Union[str, Path]] = None) -> Config:
    """
    Factory function for creating configuration instances
    
    Args:
        cli_args: Command line arguments
        config_file: Configuration file path
        
    Returns:
        Configured Config instance
    """
    return Config(cli_args=cli_args, config_file=config_file)


# For backward compatibility with existing code
def get_config(*args, **kwargs) -> Config:
    """Backward compatibility function"""
    return create_config(*args, **kwargs)


def create_unified_config(args=None) -> Config:
    """Backward compatibility function for existing main.py"""
    cli_args = vars(args) if args else {}
    return create_config(cli_args=cli_args)


