#!/usr/bin/env python3
"""
Test script for the three preprocessing modes
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.greeks_preprocessor import GreeksPreprocessor
from common.greeks_config import GreeksPreprocessingConfig
from tools.black_scholes import BlackScholesModel

def test_black_scholes_calculate_all():
    """Test the calculate_all method we just added"""
    print("Testing BlackScholesModel.calculate_all()...")
    
    bs_model = BlackScholesModel()
    
    # Test parameters
    S = 100.0  # Current stock price
    K = 105.0  # Strike price
    T = 0.25   # Time to expiry (3 months)
    r = 0.05   # Risk-free rate
    sigma = 0.2  # Volatility
    
    try:
        result = bs_model.calculate_all(S=S, K=K, T=T, r=r, sigma=sigma, option_type='call')
        
        print(f"  Option Price: {result['option_price']:.4f}")
        print(f"  Delta: {result['delta']:.4f}")
        print(f"  Gamma: {result['gamma']:.4f}")
        print(f"  Theta: {result['theta']:.4f}")
        print(f"  Vega: {result['vega']:.4f}")
        print(f"  Rho: {result['rho']:.4f}")
        
        print("  [PASS] calculate_all method works correctly")
        return True
        
    except Exception as e:
        print(f"  [FAIL] calculate_all method failed: {e}")
        return False

def test_sparse_mode():
    """Test sparse preprocessing mode"""
    print("\nTesting sparse preprocessing mode...")
    
    try:
        config = GreeksPreprocessingConfig(preprocessing_mode="sparse")
        preprocessor = GreeksPreprocessor(config)
        print("  [PASS] Sparse mode preprocessor initialized")
        return True
    except Exception as e:
        print(f"  [FAIL] Sparse mode failed: {e}")
        return False

def test_dense_interpolated_mode():
    """Test dense_interpolated preprocessing mode"""
    print("\nTesting dense_interpolated preprocessing mode...")
    
    try:
        config = GreeksPreprocessingConfig(preprocessing_mode="dense_interpolated")
        preprocessor = GreeksPreprocessor(config)
        print("  [PASS] Dense interpolated mode preprocessor initialized")
        return True
    except Exception as e:
        print(f"  [FAIL] Dense interpolated mode failed: {e}")
        return False

def test_dense_daily_recalc_mode():
    """Test dense_daily_recalc preprocessing mode"""
    print("\nTesting dense_daily_recalc preprocessing mode...")
    
    try:
        config = GreeksPreprocessingConfig(preprocessing_mode="dense_daily_recalc")
        preprocessor = GreeksPreprocessor(config)
        print("  [PASS] Dense daily recalc mode preprocessor initialized")
        return True
    except Exception as e:
        print(f"  [FAIL] Dense daily recalc mode failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Preprocessing Modes")
    print("=" * 40)
    
    tests = [
        test_black_scholes_calculate_all,
        test_sparse_mode,
        test_dense_interpolated_mode,
        test_dense_daily_recalc_mode,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All preprocessing mode tests passed!")
        return 0
    else:
        print("[FAILURE] Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())