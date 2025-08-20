"""
Custom collate functions for PyTorch DataLoader
"""

import torch
from typing import List
from .interfaces import DataResult

def data_result_collate_fn(batch: List[DataResult]) -> DataResult:
    """
    Custom collate function for DataResult objects
    
    Args:
        batch: List of DataResult objects
        
    Returns:
        Batched DataResult
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # Stack prices and holdings
    prices = torch.stack([item.prices for item in batch])
    holdings = torch.stack([item.holdings for item in batch])
    
    # Merge metadata
    merged_metadata = {
        'batch_size': len(batch),
        'dataset_type': batch[0].metadata.get('dataset_type', 'unknown')
    }
    
    # Add any common metadata keys
    for key in batch[0].metadata:
        if key not in ['dataset_type']:
            values = [item.metadata.get(key) for item in batch]
            # Only include if all items have the same value
            if all(v == values[0] for v in values):
                merged_metadata[key] = values[0]
            else:
                merged_metadata[f'{key}_list'] = values
    
    return DataResult(prices, holdings, merged_metadata)