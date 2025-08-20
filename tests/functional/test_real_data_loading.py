"""
Test real data loading functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DeltaHedgeDataset, check_delta_availability

def test_real_data_loading():
    """Test basic real data loading"""
    if not check_delta_availability():
        print("Delta hedge not available, skipping test")
        return
        
    print("Testing real data loading (limited test)...")
    
    try:
        # Create dataset with real data loading
        dataset = DeltaHedgeDataset(
            weekly_positions=['3CN5'],
            start_date='2024-01-01',
            end_date='2024-01-31'  # Limited to January for testing
        )
        
        print(f"Dataset created successfully!")
        print(f"Number of days loaded: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            print(f"First sample shape: {sample.shape}")
            print(f"First sample date: {dataset.get_date(0)}")
            print(f"Sequence lengths: {dataset.get_all_sequence_lengths()[:5]}...")  # First 5
            
    except Exception as e:
        print(f"Real data loading test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_data_loading()