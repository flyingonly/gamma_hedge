"""
Core module for Gamma Hedge system

This module provides the foundational components:
- Unified configuration management
- Standard data exchange interfaces
"""

from .config import Config, create_config
from .interfaces import TrainingData, DataProvider, TrainingPhase
from .exceptions import (
    GammaHedgeException, ConfigurationError, DataProviderError, 
    ModelError, TrainingError, MissingDependencyError
)

__all__ = [
    # Configuration system
    'Config', 
    'create_config',
    
    # Data interfaces
    'TrainingData', 
    'DataProvider', 
    'TrainingPhase',
    
    # Exceptions
    'GammaHedgeException',
    'ConfigurationError', 
    'DataProviderError',
    'ModelError',
    'TrainingError',
    'MissingDependencyError'
]