"""
Test configuration and utilities
"""

import torch
import numpy as np
from typing import Dict, Any
import tempfile
import os

# Test data generators
def create_mock_market_data(n_samples=10, n_timesteps=50, n_assets=3):
    """Create mock market data for testing"""
    torch.manual_seed(42)  # Reproducible tests
    
    prices = torch.randn(n_samples, n_timesteps, n_assets) * 0.1 + 100
    prices = torch.cumsum(prices, dim=1)  # Cumulative to simulate price paths
    prices = torch.clamp(prices, min=10)  # Ensure positive prices
    
    holdings = torch.randn(n_samples, n_timesteps, n_assets)
    
    return prices, holdings

def create_mock_option_positions():
    """Create mock option positions for testing"""
    return {
        '3CN5/CALL_111.0': 1.0,
        '3CN5/PUT_111.5': -0.5,
        '3IN5/CALL_106.5': 2.0,
    }

def create_temp_data_file(data, filename='test_data.npz'):
    """Create temporary data file for testing"""
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, filename)
    np.savez(filepath, **data)
    return filepath

# Test configuration class
class TestConfig:
    """Configuration for tests"""
    
    def __init__(self):
        self.device = 'cpu'  # Always use CPU for tests
        self.batch_size = 4
        self.n_samples = 10
        self.n_timesteps = 20
        self.n_assets = 3
        self.learning_rate = 1e-3
        self.timeout = 60  # 1 minute per test
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'device': self.device,
            'batch_size': self.batch_size,
            'n_samples': self.n_samples,
            'n_timesteps': self.n_timesteps,
            'n_assets': self.n_assets,
            'learning_rate': self.learning_rate,
        }

# Global test config instance
TEST_CONFIG = TestConfig()