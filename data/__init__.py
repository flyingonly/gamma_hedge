"""
Unified Data Processing Module for Gamma Hedge

This module provides the unified data processing interface, consolidating
all data loading, processing, and portfolio management functionality.
"""

# Core unified interfaces (recommended)
from .loaders import (
    UnifiedDataProvider, 
    create_unified_data_provider,
    MarketSimulationDataset,
    PrecomputedGreeksDataset
)
from .portfolio import PortfolioManager, create_portfolio_manager
from .processors import TimeSeriesProcessor, DataValidator, FormatConverter

# Backward compatibility imports
from .market_simulator import MarketSimulator
from .loaders import create_data_loader, create_delta_data_loader, check_delta_availability

__all__ = [
    # New unified interfaces (preferred)
    'UnifiedDataProvider',
    'create_unified_data_provider', 
    'PortfolioManager',
    'create_portfolio_manager',
    'MarketSimulationDataset',
    'PrecomputedGreeksDataset',
    'TimeSeriesProcessor',
    'DataValidator',
    'FormatConverter',
    
    # Backward compatibility
    'MarketSimulator',
    'create_data_loader',
    'create_delta_data_loader',
    'check_delta_availability'
]