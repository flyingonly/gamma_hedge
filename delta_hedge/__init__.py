"""
Delta Hedge Module
==================

This module provides delta hedging weight calculation functionality for options portfolios.
Integrates with the existing policy learning framework to provide real market-based hedging weights.

Main Components:
- data_loader: Load and preprocess underlying and options data
- delta_calculator: Core delta calculation engine
- weight_generator: Generate weights for policy learning integration
- config: Configuration management
"""

from .config import DeltaHedgeConfig
from .data_loader import OptionsDataLoader
from .delta_calculator import DeltaCalculator
from .weight_generator import WeightGenerator

__version__ = "1.0.0"
__all__ = [
    "DeltaHedgeConfig",
    "OptionsDataLoader", 
    "DeltaCalculator",
    "WeightGenerator"
]