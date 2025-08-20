"""
Test the new dynamic DeltaHedgeDataset with variable-length sequences
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DeltaHedgeDataset, check_delta_availability
from data.data_types import MarketData

def test_dynamic_delta_dataset():
    """Test the dynamic DeltaHedgeDataset with real data structure"""
    if not check_delta_availability():
        print("Delta hedge not available, skipping test")
        return
        
    print("Testing dynamic DeltaHedgeDataset...")
    
    # Create dataset with multiple weekly positions
    dataset = DeltaHedgeDataset(
        weekly_positions={
            '3CN5': 2.0,   # Long 2 contracts
            '3IN5': -1.0   # Short 1 contract
        }
    )
    
    print(f"  Dataset length (days): {len(dataset)}")
    print(f"  Number of assets: {dataset.n_assets}")
    print(f"  Underlying codes: {dataset.underlying_codes}")
    print(f"  Weekly positions: {dataset.weekly_positions}")
    
    # Test variable sequence lengths
    seq_lengths = dataset.get_all_sequence_lengths()
    print(f"  Sequence lengths per day: {seq_lengths}")
    print(f"  Min sequence length: {min(seq_lengths)}")
    print(f"  Max sequence length: {max(seq_lengths)}")
    
    # Test individual samples
    for i in range(min(3, len(dataset))):  # Test first 3 days
        sample = dataset[i]
        date = dataset.get_date(i)
        seq_len = dataset.get_sequence_length(i)
        
        print(f"\n  Day {i} ({date}):")
        print(f"    Sequence length: {seq_len}")
        print(f"    Data shape: {sample.shape}")
        print(f"    Metadata: {sample.metadata}")
        
        # Verify data consistency
        assert isinstance(sample, MarketData), f"Expected MarketData, got {type(sample)}"
        assert sample.shape[0] == seq_len, f"Shape mismatch: {sample.shape[0]} vs {seq_len}"
        assert sample.shape[1] == dataset.n_assets, f"Asset count mismatch"
    
    print("\nPASS: Dynamic DeltaHedgeDataset tests passed!")

def test_dataset_with_date_range():
    """Test dataset creation with date range"""
    if not check_delta_availability():
        print("Delta hedge not available, skipping test")
        return
        
    print("\nTesting DeltaHedgeDataset with date range...")
    
    # Create dataset with date range (will use simulation data)
    dataset = DeltaHedgeDataset(
        weekly_positions=['3CN5'],
        start_date='2024-01-01',
        end_date='2024-01-05'
    )
    
    print(f"  Dataset length: {len(dataset)}")
    print(f"  Date range simulation working: {'PASS' if len(dataset) > 0 else 'FAIL'}")

def test_single_weekly_position():
    """Test with single weekly position"""
    if not check_delta_availability():
        print("Delta hedge not available, skipping test")
        return
        
    print("\nTesting single weekly position...")
    
    dataset = DeltaHedgeDataset(weekly_positions=['3CN5'])
    
    print(f"  Single position dataset length: {len(dataset)}")
    print(f"  Underlying codes: {dataset.underlying_codes}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"  First sample shape: {sample.shape}")
        print(f"  Has metadata: {'date' in sample.metadata}")

if __name__ == "__main__":
    print("=" * 60)
    print("DYNAMIC DELTA HEDGE DATASET TESTS")
    print("=" * 60)
    
    test_dynamic_delta_dataset()
    test_dataset_with_date_range() 
    test_single_weekly_position()
    
    print("\n" + "=" * 60)
    print("All dynamic delta tests completed!")
    print("=" * 60)