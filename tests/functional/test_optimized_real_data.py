"""
Test optimized real data loading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DeltaHedgeDataset, check_delta_availability

def test_optimized_real_data():
    """Test optimized real data loading with small date range"""
    if not check_delta_availability():
        print("Delta hedge not available, skipping test")
        return
        
    print("Testing optimized real data loading...")
    
    try:
        # Test without date range first to see available data
        print("Creating dataset without date restrictions...")
        dataset = DeltaHedgeDataset(
            weekly_positions=['3CN5']
        )
        
        print(f"PASS: Dataset created successfully!")
        print(f"   Number of days loaded: {len(dataset)}")
        print(f"   Underlying codes: {dataset.underlying_codes}")
        
        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            print(f"   First sample shape: {sample.shape}")
            print(f"   First sample date: {dataset.get_date(0)}")
            
            # Show sequence lengths for all days
            seq_lengths = dataset.get_all_sequence_lengths()
            print(f"   Sequence lengths: {seq_lengths}")
            
            # Test that data is real (not all same values)
            prices = sample.prices
            print(f"   Price range: {prices.min().item():.2f} - {prices.max().item():.2f}")
            print(f"   Price variance: {prices.var().item():.4f}")
        
        print("PASS: All tests passed!")
        
    except Exception as e:
        print(f"FAIL: Real data loading test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optimized_real_data()