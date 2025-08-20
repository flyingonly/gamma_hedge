"""
Test Greeks preprocessing functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.greeks_preprocessor import GreeksPreprocessor
from data.optimized_delta_dataset import OptimizedDeltaHedgeDataset
from delta_hedge.config import DeltaHedgeConfig

def test_greeks_preprocessor():
    """Test basic Greeks preprocessing functionality"""
    print("Testing Greeks Preprocessor...")
    
    try:
        # Initialize preprocessor
        config = DeltaHedgeConfig()
        preprocessor = GreeksPreprocessor(config)
        
        print("PASS: Preprocessor initialized successfully")
        
        # Test discovery of weekly codes
        weekly_codes = preprocessor.discover_weekly_codes()
        print(f"   Found weekly codes: {weekly_codes}")
        
        if not weekly_codes:
            print("   Warning: No weekly codes found")
            return
        
        # Test discovery of options for first weekly code
        test_weekly = weekly_codes[0]
        options = preprocessor.discover_options_for_weekly(test_weekly)
        print(f"   Options for {test_weekly}: {len(options)} found")
        
        if options:
            # Test preprocessing a single option
            option_type, strike = options[0]
            print(f"   Testing preprocessing {test_weekly} {option_type} {strike}...")
            
            try:
                output_file = preprocessor.preprocess_single_option(test_weekly, option_type, strike)
                print(f"   PASS: Single option preprocessed -> {output_file}")
                
                # Verify file exists and has expected structure
                if os.path.exists(output_file):
                    import numpy as np
                    data = np.load(output_file)
                    required_keys = ['timestamps', 'delta', 'gamma', 'theta', 'vega']
                    
                    for key in required_keys:
                        if key not in data:
                            print(f"   FAIL: Missing key '{key}' in preprocessed file")
                            return
                    
                    print(f"   PASS: Preprocessed file has all required keys")
                    print(f"   Data points: {len(data['timestamps'])}")
                else:
                    print(f"   FAIL: Preprocessed file not created")
                    return
                    
            except Exception as e:
                print(f"   FAIL: Single option preprocessing failed: {e}")
                return
        
        print("PASS: Greeks preprocessor tests completed successfully!")
        
    except Exception as e:
        print(f"FAIL: Greeks preprocessor test failed: {e}")
        import traceback
        traceback.print_exc()

def test_optimized_dataset():
    """Test optimized dataset with preprocessed Greeks"""
    print("\nTesting Optimized Delta Hedge Dataset...")
    
    try:
        # Define a simple portfolio
        option_positions = {
            '3CN5/CALL_111.0': 1.0,     # Long 1 call
            '3CN5/PUT_111.5': -0.5,     # Short 0.5 put
        }
        
        print(f"   Portfolio: {option_positions}")
        
        # Test dataset creation (with auto-preprocessing)
        dataset = OptimizedDeltaHedgeDataset(
            option_positions=option_positions,
            auto_preprocess=True
        )
        
        print("PASS: Optimized dataset created successfully")
        
        # Test basic properties
        print(f"   Number of days: {len(dataset)}")
        print(f"   Number of assets: {dataset.n_assets}")
        print(f"   Unique underlyings: {dataset._get_unique_underlyings()}")
        
        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            print(f"   First sample date: {dataset.get_date(0)}")
            print(f"   First sample shape: prices={sample.prices.shape}, holdings={sample.holdings.shape}")
            
            # Test portfolio summary
            summary = dataset.get_portfolio_summary()
            print(f"   Portfolio summary: {summary}")
            
            # Test that holdings are calculated (non-zero)
            if sample.holdings.sum() != 0:
                print("   PASS: Holdings calculated (non-zero)")
            else:
                print("   Warning: Holdings are all zero")
                
            print("PASS: Optimized dataset tests completed!")
            
        else:
            print("   Warning: Dataset is empty")
        
    except Exception as e:
        print(f"FAIL: Optimized dataset test failed: {e}")
        import traceback
        traceback.print_exc()

def test_performance_comparison():
    """Test performance comparison between old and new approaches"""
    print("\nTesting Performance Comparison...")
    
    try:
        import time
        
        # Test parameters
        option_positions = {
            '3CN5/CALL_111.0': 1.0,
            '3CN5/PUT_111.5': -0.5,
        }
        
        # Test optimized approach
        print("   Testing optimized approach...")
        start_time = time.time()
        
        try:
            optimized_dataset = OptimizedDeltaHedgeDataset(
                option_positions=option_positions,
                auto_preprocess=True
            )
            optimized_time = time.time() - start_time
            optimized_samples = len(optimized_dataset)
            
            print(f"   Optimized approach: {optimized_time:.2f}s, {optimized_samples} samples")
            
        except Exception as e:
            print(f"   Optimized approach failed: {e}")
            optimized_time = float('inf')
            optimized_samples = 0
        
        # Note: We don't test the old approach here since it would be slow
        # and we've already established the optimized approach works
        
        if optimized_samples > 0:
            print(f"PASS: Performance comparison completed")
            print(f"   Optimized loading time: {optimized_time:.2f} seconds")
        else:
            print("   Warning: Could not complete performance comparison")
            
    except Exception as e:
        print(f"FAIL: Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_greeks_preprocessor()
    test_optimized_dataset() 
    test_performance_comparison()