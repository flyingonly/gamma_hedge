#!/usr/bin/env python3
"""
Test script for variable-length collate function
"""

import sys
import os
import torch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from common.collate import variable_length_collate_fn
from common.interfaces import DataResult

def test_variable_length_collate():
    """Test the variable-length collate function"""
    
    print("=" * 60)
    print("VARIABLE LENGTH COLLATE FUNCTION TEST")
    print("=" * 60)
    
    # Create test data with different sequence lengths
    # Simulate daily-aligned sequences of different lengths
    sequences = [
        # Day 1: 107 data points
        DataResult(
            prices=torch.randn(107, 1),
            holdings=torch.randn(107, 1),
            metadata={'trading_date': '2025-01-01', 'sequence_length': 107}
        ),
        # Day 2: 168 data points
        DataResult(
            prices=torch.randn(168, 1),
            holdings=torch.randn(168, 1),
            metadata={'trading_date': '2025-01-02', 'sequence_length': 168}
        ),
        # Day 3: 95 data points
        DataResult(
            prices=torch.randn(95, 1),
            holdings=torch.randn(95, 1),
            metadata={'trading_date': '2025-01-03', 'sequence_length': 95}
        )
    ]
    
    print(f"[INFO] Input sequences:")
    for i, seq in enumerate(sequences):
        print(f"  Sequence {i+1}: {seq.prices.shape} - {seq.metadata['trading_date']}")
    
    try:
        # Test variable-length collate function
        batched_result = variable_length_collate_fn(sequences)
        
        print(f"\n[INFO] Batched result:")
        print(f"  Prices shape: {batched_result.prices.shape}")
        print(f"  Holdings shape: {batched_result.holdings.shape}")
        print(f"  Is variable length: {batched_result.metadata.get('is_variable_length', False)}")
        print(f"  Original lengths: {batched_result.metadata.get('original_lengths', [])}")
        print(f"  Max sequence length: {batched_result.metadata.get('max_sequence_length', 0)}")
        
        # Test attention mask
        attention_mask = batched_result.metadata.get('attention_mask')
        if attention_mask is not None:
            print(f"  Attention mask shape: {attention_mask.shape}")
            print(f"  Attention mask dtype: {attention_mask.dtype}")
            
            # Verify mask correctness
            original_lengths = batched_result.metadata['original_lengths']
            for batch_idx, orig_len in enumerate(original_lengths):
                valid_positions = attention_mask[batch_idx, :orig_len].sum().item()
                padding_positions = attention_mask[batch_idx, orig_len:].sum().item()
                print(f"  Batch {batch_idx}: {valid_positions} valid, {padding_positions} padding positions")
                
                # Verify all valid positions are marked as 1
                assert valid_positions == orig_len, f"Expected {orig_len} valid positions, got {valid_positions}"
                assert padding_positions == 0, f"Expected 0 padding positions to be marked valid, got {padding_positions}"
        
        print(f"\n[PASS] Variable-length collate function test passed!")
        print(f"[PASS] Successfully batched sequences of lengths {[seq.prices.shape[0] for seq in sequences]}")
        print(f"[PASS] Attention mask correctly identifies valid vs padding positions")
        
        # Test that this would work with DataLoader
        from torch.utils.data import DataLoader, Dataset
        
        class TestDataset(Dataset):
            def __init__(self, sequences):
                self.sequences = sequences
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                return self.sequences[idx]
        
        dataset = TestDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=3, collate_fn=variable_length_collate_fn)
        
        # Test one batch
        for batch in dataloader:
            print(f"\n[INFO] DataLoader batch test:")
            print(f"  Batch prices shape: {batch.prices.shape}")
            print(f"  Batch holdings shape: {batch.holdings.shape}")
            print(f"  Original lengths: {batch.metadata.get('original_lengths', [])}")
            break
        
        print(f"\n[PASS] DataLoader integration test passed!")
        
        print("\n" + "=" * 60)
        print("VARIABLE LENGTH COLLATE TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"[FAIL] Variable-length collate test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_variable_length_collate()