"""
Delta Hedge System Testing Suite
===============================

Comprehensive testing for delta hedging calculations and weight generation.
Validates accuracy, performance, and integration capabilities.
"""

import numpy as np
import pandas as pd
import logging
import time
import sys
import os
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# Try to import torch, handle if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock torch tensors for testing
    class MockTensor:
        def __init__(self, data, dtype=None):
            self.data = np.array(data)
            self.shape = self.data.shape
            self.dtype = dtype or 'float32'
        
        def numpy(self):
            return self.data
    
    class MockTorch:
        float32 = 'float32'
        
        @staticmethod
        def tensor(data, dtype=None):
            return MockTensor(data, dtype)
        
        @staticmethod
        def stack(tensors):
            stacked_data = np.stack([t.data for t in tensors])
            return MockTensor(stacked_data)
        
        @staticmethod
        def randn_like(tensor):
            return MockTensor(np.random.randn(*tensor.shape))
        
        @staticmethod
        def cat(tensors, dim=-1):
            concatenated = np.concatenate([t.data for t in tensors], axis=dim)
            return MockTensor(concatenated)
        
        @staticmethod
        def full_like(tensor, value):
            return MockTensor(np.full_like(tensor.data, value))
    
    torch = MockTorch()

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delta_hedge.config import DeltaHedgeConfig
from delta_hedge.data_loader import OptionsDataLoader
from delta_hedge.bs_integration import EnhancedBlackScholesCalculator
from delta_hedge.delta_calculator import DeltaCalculator
from delta_hedge.weight_generator import WeightGenerator, create_default_weight_generator
from tools.black_scholes import BlackScholesModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeltaSystemTester:
    """
    Comprehensive testing suite for delta hedge system.
    """
    
    def __init__(self):
        """Initialize test suite"""
        self.config = DeltaHedgeConfig()
        self.test_results = {}
        logger.info("Delta system tester initialized")
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all test suites and return results.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Starting comprehensive delta system testing")
        
        # Test suites
        test_suites = [
            ("black_scholes_accuracy", self.test_black_scholes_accuracy),
            ("delta_calculation_accuracy", self.test_delta_calculation_accuracy),
            ("portfolio_delta_aggregation", self.test_portfolio_delta_aggregation),
            ("weight_generation_format", self.test_weight_generation_format),
            ("system_integration", self.test_system_integration),
            ("performance_benchmarks", self.test_performance_benchmarks),
            ("edge_cases", self.test_edge_cases),
            ("data_compatibility", self.test_data_compatibility)
        ]
        
        for test_name, test_func in test_suites:
            try:
                logger.info(f"Running test suite: {test_name}")
                result = test_func()
                self.test_results[test_name] = result
                logger.info(f"Test suite {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test suite {test_name} failed with exception: {e}")
                self.test_results[test_name] = False
        
        # Summary
        passed_tests = sum(self.test_results.values())
        total_tests = len(self.test_results)
        logger.info(f"Testing complete: {passed_tests}/{total_tests} test suites passed")
        
        return self.test_results
    
    def test_black_scholes_accuracy(self) -> bool:
        """Test Black-Scholes calculation accuracy against known values"""
        logger.info("Testing Black-Scholes calculation accuracy")
        
        # Test cases with corrected analytical results
        test_cases = [
            # (S, K, T, r, sigma, option_type, expected_price, expected_delta, tolerance)
            (100, 100, 1.0, 0.05, 0.2, 'call', 10.451, 0.637, 0.01),
            (100, 100, 1.0, 0.05, 0.2, 'put', 5.574, -0.363, 0.01),
            (110, 100, 0.5, 0.03, 0.15, 'call', 12.340, 0.863, 0.01),
            (90, 100, 0.25, 0.02, 0.3, 'put', 11.634, -0.724, 0.01),
        ]
        
        bs_calculator = EnhancedBlackScholesCalculator(self.config)
        all_passed = True
        
        for S, K, T, r, sigma, option_type, exp_price, exp_delta, tolerance in test_cases:
            # Calculate price and Greeks
            price = bs_calculator.calculate_option_price(S, K, T, r, sigma, option_type)[0]
            greeks = bs_calculator.calculate_greeks_batch(S, K, T, r, sigma, option_type)
            delta = greeks['delta'][0]
            
            # Check accuracy
            price_error = abs(price - exp_price) / exp_price
            delta_error = abs(delta - exp_delta) / abs(exp_delta)
            
            if price_error > tolerance or delta_error > tolerance:
                logger.error(f"BS accuracy test failed for {option_type}: price_error={price_error:.4f}, delta_error={delta_error:.4f}")
                all_passed = False
            else:
                logger.debug(f"BS test passed for {option_type}: price={price:.2f}, delta={delta:.3f}")
        
        return all_passed
    
    def test_delta_calculation_accuracy(self) -> bool:
        """Test delta calculation accuracy for various scenarios"""
        logger.info("Testing delta calculation accuracy")
        
        delta_calculator = DeltaCalculator(self.config)
        
        # Test single option delta calculation
        test_passed = True
        
        # Test case 1: At-the-money call option
        delta_call = delta_calculator.calculate_single_option_delta(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call'
        )
        if not (0.6 <= delta_call <= 0.7):
            logger.error(f"ATM call delta out of expected range: {delta_call}")
            test_passed = False
        
        # Test case 2: At-the-money put option
        delta_put = delta_calculator.calculate_single_option_delta(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='put'
        )
        if not (-0.5 <= delta_put <= -0.3):
            logger.error(f"ATM put delta out of expected range: {delta_put}")
            test_passed = False
        
        # Test case 3: Put-call parity for delta
        # For European options: delta_call - delta_put = e^(-q*T) where q is dividend yield
        # With q=0: delta_call - delta_put = 1, so delta_call + delta_put = 1 - 2*delta_put
        # Actually, let's check call - put = 1 (more standard)
        delta_diff = delta_call - delta_put  
        if abs(delta_diff - 1.0) > 0.01:
            logger.error(f"Put-call delta parity violated: call_delta - put_delta = {delta_diff}, expected ~1.0")
            test_passed = False
        
        return test_passed
    
    def test_portfolio_delta_aggregation(self) -> bool:
        """Test portfolio-level delta aggregation"""
        logger.info("Testing portfolio delta aggregation")
        
        delta_calculator = DeltaCalculator(self.config)
        
        # Create synthetic portfolio
        portfolio_positions = {
            'CALL_100': 2.0,   # Long 2 ATM calls
            'PUT_100': -1.0,   # Short 1 ATM put
            'CALL_110': -0.5   # Short 0.5 OTM calls
        }
        
        # Create synthetic options data
        underlying_price = 105.0
        options_data = {
            'CALL_100': {
                'strikes': np.array([100.0]),
                'option_types': ['call'],
                'market_prices': np.array([8.0]),
                'implied_volatilities': np.array([0.2])
            },
            'PUT_100': {
                'strikes': np.array([100.0]),
                'option_types': ['put'],
                'market_prices': np.array([3.0]),
                'implied_volatilities': np.array([0.2])
            },
            'CALL_110': {
                'strikes': np.array([110.0]),
                'option_types': ['call'],
                'market_prices': np.array([2.0]),
                'implied_volatilities': np.array([0.18])
            }
        }
        
        # Create sample contract info for testing
        test_contract_info = {
            'expiry_date': '2025-12-31',  # Future expiry for testing
            'underlying': 'USU5',
            'product': 'US'
        }
        
        # Calculate portfolio delta
        portfolio_metrics = delta_calculator.calculate_portfolio_delta(
            portfolio_positions=portfolio_positions,
            options_data=options_data,
            underlying_price=underlying_price,
            timestamp=datetime.now(),
            contract_info=test_contract_info
        )
        
        portfolio_delta = portfolio_metrics['portfolio_delta']
        
        # Validate results
        if not isinstance(portfolio_delta, float):
            logger.error("Portfolio delta should be a float")
            return False
        
        # Portfolio delta should be positive (net long calls)
        if portfolio_delta <= 0:
            logger.error(f"Expected positive portfolio delta but got: {portfolio_delta}")
            return False
        
        logger.info(f"Portfolio delta calculation passed: {portfolio_delta:.3f}")
        return True
    
    def test_weight_generation_format(self) -> bool:
        """Test weight generation output format and structure"""
        logger.info("Testing weight generation format")
        
        # Test configuration
        underlying_codes = ['USU5', 'FVU5']
        weekly_codes = ['3CN5']
        
        try:
            generator = create_default_weight_generator(underlying_codes, weekly_codes)
            
            # Generate test batch
            batch_data = generator.generate_delta_weights_batch(
                underlying_codes=underlying_codes,
                weekly_codes=weekly_codes,
                batch_size=3,
                sequence_length=5
            )
            
            # Validate output structure
            expected_keys = ['prices', 'holdings', 'delta_weights', 'delta_info', 'risk_metrics']
            if set(batch_data.keys()) != set(expected_keys):
                logger.error(f"Expected keys {expected_keys}, got {list(batch_data.keys())}")
                return False
            
            # Validate tensor shapes
            batch_size, seq_len, n_assets = 3, 5, 2
            expected_shapes = {
                'prices': (batch_size, seq_len, n_assets),
                'holdings': (batch_size, seq_len, n_assets),
                'delta_weights': (batch_size, seq_len, n_assets),
                'delta_info': (batch_size, seq_len, 3),
                'risk_metrics': (batch_size, seq_len, 4)
            }
            
            for key, expected_shape in expected_shapes.items():
                actual_shape = batch_data[key].shape
                if actual_shape != expected_shape:
                    logger.error(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
                    return False
            
            # Validate tensor types
            for key, tensor in batch_data.items():
                if TORCH_AVAILABLE:
                    if not isinstance(tensor, torch.Tensor):
                        logger.error(f"{key} should be a torch.Tensor")
                        return False
                    if tensor.dtype != torch.float32:
                        logger.error(f"{key} should have dtype float32")
                        return False
                else:
                    if not hasattr(tensor, 'shape') or not hasattr(tensor, 'data'):
                        logger.error(f"{key} should be tensor-like")
                        return False
            
            logger.info("Weight generation format test passed")
            return True
            
        except Exception as e:
            logger.error(f"Weight generation test failed: {e}")
            return False
    
    def test_system_integration(self) -> bool:
        """Test integration between all system components"""
        logger.info("Testing system integration")
        
        try:
            # Test end-to-end workflow
            underlying_codes = ['USU5']
            weekly_codes = ['3CN5']
            
            # Create weight generator (integrates all components)
            generator = create_default_weight_generator(underlying_codes, weekly_codes)
            
            # Test market simulator compatibility
            prices, holdings = generator.create_market_simulator_compatible_data(
                underlying_codes=underlying_codes,
                weekly_codes=weekly_codes,
                n_paths=2,
                n_timesteps=3
            )
            
            # Validate output format matches market simulator expectations
            if prices.shape != (2, 3, 1) or holdings.shape[0:2] != (2, 3):
                logger.error(f"Market simulator compatibility failed: prices={prices.shape}, holdings={holdings.shape}")
                return False
            
            # Test feature dimensions
            dimensions = generator.get_feature_dimensions(underlying_codes, weekly_codes)
            required_dims = ['n_assets', 'price_dim', 'holdings_dim', 'delta_weights_dim']
            
            if not all(dim in dimensions for dim in required_dims):
                logger.error(f"Missing required dimensions: {dimensions}")
                return False
            
            logger.info("System integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"System integration test failed: {e}")
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks for the system"""
        logger.info("Testing performance benchmarks")
        
        try:
            generator = create_default_weight_generator(['USU5'], ['3CN5'])
            
            # Benchmark weight generation
            start_time = time.time()
            batch_data = generator.generate_delta_weights_batch(
                underlying_codes=['USU5'],
                weekly_codes=['3CN5'],
                batch_size=10,
                sequence_length=20
            )
            generation_time = time.time() - start_time
            
            # Performance criteria
            max_generation_time = 5.0  # 5 seconds for 10 paths of 20 timesteps
            
            if generation_time > max_generation_time:
                logger.error(f"Weight generation too slow: {generation_time:.2f}s > {max_generation_time}s")
                return False
            
            logger.info(f"Performance benchmark passed: generation time {generation_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            return False
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling"""
        logger.info("Testing edge cases")
        
        delta_calculator = DeltaCalculator(self.config)
        
        # Test edge case 1: Zero time to expiry
        try:
            delta = delta_calculator.calculate_single_option_delta(
                S=100, K=100, T=0.001, r=0.05, sigma=0.2, option_type='call'
            )
            # Should not crash and return reasonable value
            if not isinstance(delta, float) or not np.isfinite(delta):
                logger.error("Invalid delta for near-expiry option")
                return False
        except Exception as e:
            logger.error(f"Failed to handle near-expiry case: {e}")
            return False
        
        # Test edge case 2: Very high volatility
        try:
            delta = delta_calculator.calculate_single_option_delta(
                S=100, K=100, T=1.0, r=0.05, sigma=2.0, option_type='call'
            )
            # Should handle high volatility gracefully
            if not isinstance(delta, float) or not np.isfinite(delta):
                logger.error("Invalid delta for high volatility")
                return False
        except Exception as e:
            logger.error(f"Failed to handle high volatility case: {e}")
            return False
        
        # Test edge case 3: Empty portfolio
        try:
            test_contract_info = {
                'expiry_date': '2025-12-31',
                'underlying': 'USU5',
                'product': 'US'
            }
            portfolio_metrics = delta_calculator.calculate_portfolio_delta(
                portfolio_positions={},
                options_data={},
                underlying_price=100.0,
                timestamp=datetime.now(),
                contract_info=test_contract_info
            )
            if portfolio_metrics['portfolio_delta'] != 0.0:
                logger.error("Empty portfolio should have zero delta")
                return False
        except Exception as e:
            logger.error(f"Failed to handle empty portfolio: {e}")
            return False
        
        logger.info("Edge cases test passed")
        return True
    
    def test_data_compatibility(self) -> bool:
        """Test compatibility with existing data formats"""
        logger.info("Testing data format compatibility")
        
        try:
            # Test that the system can handle the expected data formats
            # from the existing csv_process pipeline
            
            # This is a basic compatibility test since we don't have
            # access to the actual data files in testing
            config = DeltaHedgeConfig()
            
            # Test configuration validity
            if not os.path.exists("csv_process"):
                logger.warning("csv_process directory not found - skipping data compatibility test")
                return True
            
            # Test data loader initialization
            try:
                data_loader = OptionsDataLoader(config)
                logger.info("Data loader initialization successful")
                return True
            except Exception as e:
                logger.error(f"Data loader initialization failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Data compatibility test failed: {e}")
            return False
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 60)
        report.append("DELTA HEDGE SYSTEM TEST REPORT")
        report.append("=" * 60)
        report.append(f"Test execution time: {datetime.now()}")
        report.append(f"Configuration: {self.config.__class__.__name__}")
        report.append("")
        
        if not self.test_results:
            report.append("No tests have been run yet.")
            return "\n".join(report)
        
        # Summary
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        report.append(f"SUMMARY: {passed}/{total} tests passed ({success_rate:.1f}%)")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            report.append(f"  {test_name:.<40} {status}")
        
        report.append("")
        
        # Recommendations
        if passed == total:
            report.append("ALL TESTS PASSED - System ready for production use")
        elif passed >= total * 0.8:
            report.append("MOST TESTS PASSED - Review failed tests before deployment")
        else:
            report.append("MULTIPLE TESTS FAILED - System needs significant fixes")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Run the complete test suite"""
    print("Starting Delta Hedge System Testing...")
    
    tester = DeltaSystemTester()
    results = tester.run_all_tests()
    
    # Generate and print report
    report = tester.generate_test_report()
    print("\n" + report)
    
    # Save report to file
    report_path = "tests/delta_hedge_system_tests/test_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nTest report saved to: {report_path}")
    
    # Return exit code based on results
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)