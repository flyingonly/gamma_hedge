"""
Test multi-underlying support in unified DeltaHedgeDataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DeltaHedgeDataset, check_delta_availability

def test_multi_underlying_specific_options():
    """Test specific options from multiple underlyings"""
    print("Testing multi-underlying support with specific options...")
    
    if not check_delta_availability():
        print("Delta hedge not available, skipping test")
        return
    
    try:
        # Portfolio with options from different underlyings
        option_positions = {
            '3CN5/CALL_111.0': 1.0,     # USU5 underlying (corn -> treasury note)
            '3CN5/PUT_111.5': -0.5,     # USU5 underlying
            '3IN5/CALL_106.5': 2.0,     # FVU5 underlying (iron ore -> 5-year note)
            '3IN5/PUT_107.0': -1.0,     # FVU5 underlying
            '3MN5/CALL_110.0': 0.5,     # TUU5 underlying (meal -> 2-year note)
            '3WN5/PUT_103.5': -0.8,     # TYU5 underlying (wheat -> 10-year note)
        }
        
        print(f"   Portfolio: {len(option_positions)} options from multiple underlyings")
        
        # Create dataset
        dataset = DeltaHedgeDataset(
            positions=option_positions,
            use_preprocessed_greeks=True,
            auto_preprocess=True
        )
        
        print("PASS: Dataset created successfully with multi-underlying options")
        
        # Check underlying mapping
        unique_underlyings = dataset._get_unique_underlyings()
        print(f"   Unique underlyings detected: {unique_underlyings}")
        
        expected_underlyings = ['FVU5', 'TUU5', 'TYU5', 'USU5']  # Sorted order
        
        if set(unique_underlyings) == set(expected_underlyings):
            print("   PASS: All expected underlyings detected")
        else:
            print(f"   Warning: Expected {expected_underlyings}, got {unique_underlyings}")
        
        print(f"   Number of days: {len(dataset)}")
        print(f"   Number of underlying assets: {dataset.n_assets}")
        
        if len(dataset) > 0:
            # Test sample data structure
            sample = dataset[0]
            print(f"   Sample shapes: prices={sample.prices.shape}, holdings={sample.holdings.shape}")
            
            # Verify that holdings are calculated for different underlyings
            holdings_sum_by_asset = sample.holdings.sum(dim=0)  # Sum over time
            non_zero_assets = (holdings_sum_by_asset.abs() > 1e-6).sum().item()
            
            print(f"   Assets with non-zero holdings: {non_zero_assets}/{dataset.n_assets}")
            
            if non_zero_assets > 1:
                print("   PASS: Multiple underlying assets have holdings")
            else:
                print("   Warning: Only one or no underlying assets have holdings")
            
            # Show portfolio summary
            summary = dataset.get_portfolio_summary()
            print(f"   Portfolio summary keys: {list(summary.keys())}")
        
        print("PASS: Multi-underlying specific options test completed!")
        
    except Exception as e:
        print(f"FAIL: Multi-underlying test failed: {e}")
        import traceback
        traceback.print_exc()

def test_multi_underlying_legacy_weekly():
    """Test legacy weekly positions with multiple underlyings"""
    print("\nTesting multi-underlying support with legacy weekly positions...")
    
    if not check_delta_availability():
        print("Delta hedge not available, skipping test")
        return
    
    try:
        # Legacy weekly positions from different underlyings
        weekly_positions = {
            '3CN5': 1.0,    # USU5 underlying
            '3IN5': -0.5,   # FVU5 underlying  
            '3MN5': 2.0,    # TUU5 underlying
        }
        
        print(f"   Weekly positions: {weekly_positions}")
        
        # Create dataset (should auto-convert to specific options)
        dataset = DeltaHedgeDataset(
            positions=weekly_positions,
            use_preprocessed_greeks=True,
            auto_preprocess=True
        )
        
        print("PASS: Dataset created successfully with legacy weekly positions")
        
        # Check that conversion happened
        print(f"   Converted to option positions: {len(dataset.option_positions)} options")
        print(f"   Option IDs: {list(dataset.option_positions.keys())}")
        
        # Check underlying diversity
        unique_underlyings = dataset._get_unique_underlyings()
        print(f"   Unique underlyings: {unique_underlyings}")
        
        if len(unique_underlyings) >= 2:
            print("   PASS: Multiple underlyings detected from legacy positions")
        else:
            print(f"   Warning: Expected multiple underlyings, got {len(unique_underlyings)}")
        
        print("PASS: Multi-underlying legacy weekly test completed!")
        
    except Exception as e:
        print(f"FAIL: Legacy weekly test failed: {e}")
        import traceback
        traceback.print_exc()

def test_mixed_position_formats():
    """Test different position format combinations"""
    print("\nTesting different position format combinations...")
    
    test_cases = [
        {
            'name': 'List format (legacy)',
            'positions': ['3CN5', '3IN5', '3MN5'],
            'expected_types': 'weekly -> options conversion'
        },
        {
            'name': 'Weekly dict format (legacy)',
            'positions': {'3CN5': 1.0, '3IN5': -0.5},
            'expected_types': 'weekly -> options conversion'
        },
        {
            'name': 'Specific options format',
            'positions': {'3CN5/CALL_111.0': 1.0, '3IN5/PUT_107.0': -0.5},
            'expected_types': 'direct options'
        }
    ]
    
    for test_case in test_cases:
        try:
            print(f"   Testing {test_case['name']}...")
            
            # Don't create full dataset to save time, just test parsing
            dataset = DeltaHedgeDataset.__new__(DeltaHedgeDataset)
            dataset.weekly_positions = {}
            dataset.option_positions = {}
            
            # Test position parsing
            dataset._parse_positions(test_case['positions'])
            
            has_weekly = bool(dataset.weekly_positions)
            has_options = bool(dataset.option_positions)
            
            print(f"     Weekly positions: {has_weekly}, Option positions: {has_options}")
            print(f"     Expected: {test_case['expected_types']}")
            
            if test_case['name'].startswith('Specific options'):
                if has_options and not has_weekly:
                    print("     PASS: Specific options parsed correctly")
                else:
                    print("     FAIL: Specific options not parsed correctly")
            else:
                if has_weekly and not has_options:
                    print("     PASS: Legacy format parsed correctly")
                else:
                    print("     FAIL: Legacy format not parsed correctly")
            
        except Exception as e:
            print(f"     FAIL: {test_case['name']} parsing failed: {e}")

def test_underlying_mapping():
    """Test underlying asset code mapping"""
    print("\nTesting underlying asset mapping...")
    
    try:
        # Create a dummy dataset to test mapping function
        dataset = DeltaHedgeDataset.__new__(DeltaHedgeDataset)
        
        test_mappings = [
            ('3CN5', 'USU5'),  # Corn -> 10-year treasury
            ('3IN5', 'FVU5'),  # Iron ore -> 5-year treasury
            ('3MN5', 'TUU5'),  # Meal -> 2-year treasury
            ('3WN5', 'TYU5'),  # Wheat -> 10-year treasury
            ('3XYZ', 'USU5'),  # Unknown -> default
        ]
        
        all_passed = True
        for weekly_code, expected_underlying in test_mappings:
            actual_underlying = dataset._map_weekly_to_underlying(weekly_code)
            
            if actual_underlying == expected_underlying:
                print(f"   {weekly_code} -> {actual_underlying}: PASS")
            else:
                print(f"   {weekly_code} -> {actual_underlying} (expected {expected_underlying}): FAIL")
                all_passed = False
        
        if all_passed:
            print("PASS: All underlying mappings correct!")
        else:
            print("FAIL: Some underlying mappings incorrect")
            
    except Exception as e:
        print(f"FAIL: Underlying mapping test failed: {e}")

if __name__ == "__main__":
    test_multi_underlying_specific_options()
    test_multi_underlying_legacy_weekly()
    test_mixed_position_formats()
    test_underlying_mapping()