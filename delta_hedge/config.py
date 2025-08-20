"""
Delta Hedge Configuration Module
===============================

Configuration management for delta hedging system.
Defines all parameters for data paths, risk management, and integration with policy learning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data path and loading configuration"""
    # Base data directories
    underlying_data_dir: str = "./csv_process/underlying_npz"
    options_data_dir: str = "./csv_process/weekly_options_data" 
    mapping_file: str = "./csv_process/weekly_options_data/weekly_mapping.json"
    
    # Available underlying assets
    available_underlyings: List[str] = None
    
    # Data processing parameters
    time_alignment_method: str = "forward_fill"  # "interpolate", "forward_fill", "backward_fill"
    missing_data_threshold: float = 0.1  # Maximum allowed missing data ratio
    
    # TODO: TEMPORARY_LOGIC - Using forward_fill as default for option price alignment
    # This temporarily fills missing option prices with the last available price
    # Should be replaced with more sophisticated pricing models in the future
    
    def __post_init__(self):
        if self.available_underlyings is None:
            self.available_underlyings = [
                "FVU5", "FVZ5", "TUU5", "TUZ5", 
                "TYU5", "TYZ5", "USU5", "USZ5"
            ]


@dataclass 
class BlackScholesConfig:
    """Black-Scholes model parameters"""
    # Risk-free rate settings
    risk_free_rate: float = 0.05  # Default 5% annual rate
    use_dynamic_rate: bool = False  # Whether to use time-varying rates
    
    # Volatility calculation
    volatility_window: int = 30  # Days for historical volatility calculation
    min_volatility: float = 0.01  # Minimum volatility (1%)
    max_volatility: float = 3.0   # Maximum volatility (300%)
    
    # Time to expiration
    min_time_to_expiry: float = 1/365  # Minimum 1 day
    business_days_per_year: int = 252
    
    # Numerical precision
    calculation_tolerance: float = 1e-8


@dataclass
class DeltaCalculationConfig:
    """Delta calculation and hedging parameters"""
    # Delta calculation method
    delta_method: str = "analytical"  # "analytical", "numerical", "market_implied"
    numerical_bump_size: float = 0.01  # For numerical delta calculation
    
    # Portfolio aggregation
    portfolio_delta_method: str = "weighted_sum"  # "weighted_sum", "risk_weighted"
    
    # Hedging parameters
    rebalance_threshold: float = 0.05  # Rebalance when delta changes by 5%
    min_hedge_notional: float = 100.0  # Minimum notional for hedging
    max_hedge_ratio: float = 10.0      # Maximum hedge ratio
    
    # Risk limits
    max_portfolio_delta: float = 1000.0  # Maximum absolute portfolio delta
    max_individual_weight: float = 0.5   # Maximum weight for single position
    
    # Expiry handling
    enable_expiry_checking: bool = True   # Enable automatic expiry date checking
    skip_expired_options: bool = True     # Skip calculations for expired options


@dataclass
class WeightGenerationConfig:
    """Weight generation for policy learning integration"""
    # Output format
    output_format: str = "pytorch"  # "pytorch", "numpy", "pandas"
    batch_size: int = 32
    sequence_length: int = 10
    
    # Weight scaling
    weight_scaling_method: str = "normalize"  # "normalize", "standardize", "minmax"
    target_weight_range: tuple = (-1.0, 1.0)  # Target range for weights
    
    # Noise and simulation
    add_weight_noise: bool = True
    weight_noise_std: float = 0.01  # 1% noise standard deviation
    
    # Integration with existing market simulator
    compatible_with_market_sim: bool = True
    asset_dimension_consistency: bool = True


@dataclass
class PolicyLearningIntegrationConfig:
    """Integration parameters with existing policy learning framework"""
    # Model integration
    extend_state_space: bool = True      # Add delta info to state space
    delta_as_constraint: bool = False    # Use delta as action constraint
    delta_in_reward: bool = True         # Include delta effectiveness in reward
    
    # Input dimensions (to be determined based on data)
    n_assets: Optional[int] = None
    n_options: Optional[int] = None
    n_timesteps: Optional[int] = None
    
    # Feature engineering
    include_greeks: bool = True          # Include Gamma, Theta, Vega
    include_risk_metrics: bool = True    # Include VaR, portfolio volatility
    normalize_features: bool = True      # Normalize all features
    
    # Training integration
    save_weights_with_checkpoints: bool = True
    weight_update_frequency: int = 1     # Update weights every N epochs


@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""
    # Computation optimization
    use_vectorized_calculations: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None  # None = auto-detect
    
    # Memory management
    batch_processing: bool = True
    max_memory_usage_gb: float = 4.0
    cache_calculations: bool = True
    
    # Logging and monitoring
    enable_performance_monitoring: bool = True
    log_calculation_times: bool = False
    save_intermediate_results: bool = False


@dataclass
class DeltaHedgeConfig:
    """Main configuration class that combines all sub-configurations"""
    
    data: DataConfig = None
    black_scholes: BlackScholesConfig = None
    delta_calculation: DeltaCalculationConfig = None
    weight_generation: WeightGenerationConfig = None
    policy_integration: PolicyLearningIntegrationConfig = None
    performance: PerformanceConfig = None
    
    # Global settings
    random_seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize all sub-configurations if not provided"""
        if self.data is None:
            self.data = DataConfig()
        if self.black_scholes is None:
            self.black_scholes = BlackScholesConfig()
        if self.delta_calculation is None:
            self.delta_calculation = DeltaCalculationConfig()
        if self.weight_generation is None:
            self.weight_generation = WeightGenerationConfig()
        if self.policy_integration is None:
            self.policy_integration = PolicyLearningIntegrationConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
    
    def validate(self) -> bool:
        """Validate configuration consistency"""
        errors = []
        
        # Validate data paths
        if not Path(self.data.underlying_data_dir).exists():
            errors.append(f"Underlying data directory not found: {self.data.underlying_data_dir}")
        if not Path(self.data.options_data_dir).exists():
            errors.append(f"Options data directory not found: {self.data.options_data_dir}")
        if not Path(self.data.mapping_file).exists():
            errors.append(f"Mapping file not found: {self.data.mapping_file}")
        
        # Validate parameter ranges
        if not (0 <= self.black_scholes.risk_free_rate <= 1):
            errors.append("Risk-free rate should be between 0 and 1")
        if not (0 < self.delta_calculation.rebalance_threshold <= 1):
            errors.append("Rebalance threshold should be between 0 and 1")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        return True
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization"""
        return {
            "data": self.data.__dict__,
            "black_scholes": self.black_scholes.__dict__,
            "delta_calculation": self.delta_calculation.__dict__,
            "weight_generation": self.weight_generation.__dict__,
            "policy_integration": self.policy_integration.__dict__,
            "performance": self.performance.__dict__,
            "random_seed": self.random_seed,
            "device": self.device,
            "verbose": self.verbose
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create configuration from dictionary"""
        data_config = DataConfig(**config_dict.get("data", {}))
        bs_config = BlackScholesConfig(**config_dict.get("black_scholes", {}))
        delta_config = DeltaCalculationConfig(**config_dict.get("delta_calculation", {}))
        weight_config = WeightGenerationConfig(**config_dict.get("weight_generation", {}))
        policy_config = PolicyLearningIntegrationConfig(**config_dict.get("policy_integration", {}))
        perf_config = PerformanceConfig(**config_dict.get("performance", {}))
        
        return cls(
            data=data_config,
            black_scholes=bs_config,
            delta_calculation=delta_config,
            weight_generation=weight_config,
            policy_integration=policy_config,
            performance=perf_config,
            random_seed=config_dict.get("random_seed", 42),
            device=config_dict.get("device", "auto"),
            verbose=config_dict.get("verbose", True)
        )


def get_default_config() -> DeltaHedgeConfig:
    """Get default configuration for delta hedging system"""
    return DeltaHedgeConfig()


def get_policy_learning_config() -> DeltaHedgeConfig:
    """Get optimized configuration for policy learning integration"""
    config = DeltaHedgeConfig()
    
    # Optimize for policy learning
    config.weight_generation.batch_size = 64
    config.weight_generation.sequence_length = 20
    config.policy_integration.extend_state_space = True
    config.policy_integration.include_greeks = True
    config.policy_integration.normalize_features = True
    
    # Performance optimizations
    config.performance.use_vectorized_calculations = True
    config.performance.parallel_processing = True
    config.performance.cache_calculations = True
    
    return config


def get_testing_config() -> DeltaHedgeConfig:
    """Get configuration optimized for testing and development"""
    config = DeltaHedgeConfig()
    
    # Smaller batches for testing
    config.weight_generation.batch_size = 4
    config.weight_generation.sequence_length = 5
    
    # Enable detailed logging
    config.performance.log_calculation_times = True
    config.performance.save_intermediate_results = True
    config.verbose = True
    
    return config