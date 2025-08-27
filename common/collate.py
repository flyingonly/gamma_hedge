"""
Custom collate functions for PyTorch DataLoader
"""

import torch
from typing import List, Tuple
from core.interfaces import DataResult

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
    
    # Stack time_features if present
    time_features = None
    if batch[0].time_features is not None:
        time_features = torch.stack([item.time_features for item in batch])
    
    # Stack history_features if present
    history_features = None
    if batch[0].history_features is not None:
        history_features = torch.stack([item.history_features for item in batch])
    
    # Stack delta_features if present
    delta_features = None
    if batch[0].delta_features is not None:
        delta_features = torch.stack([item.delta_features for item in batch])
    
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
    
    return DataResult(
        prices=prices, 
        holdings=holdings, 
        metadata=merged_metadata,
        time_features=time_features,
        history_features=history_features,
        delta_features=delta_features
    )

def create_attention_mask(padded_tensor: torch.Tensor, original_lengths: List[int]) -> torch.Tensor:
    """
    Create attention mask for padded sequences
    
    Args:
        padded_tensor: Padded tensor with shape [batch_size, max_seq_len, features]
        original_lengths: List of original sequence lengths for each batch item
        
    Returns:
        Attention mask with shape [batch_size, max_seq_len] where 1=valid, 0=padding
    """
    batch_size, max_seq_len = padded_tensor.shape[:2]
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    
    for batch_idx, length in enumerate(original_lengths):
        mask[batch_idx, :length] = 1
    
    return mask

def variable_length_collate_fn(batch: List[DataResult]) -> DataResult:
    """
    Custom collate function for variable-length DataResult objects (daily-aligned sequences)
    
    This function handles sequences of different lengths by padding them to the same length
    and creating attention masks to identify valid vs padded positions.
    
    Args:
        batch: List of DataResult objects with potentially different sequence lengths
        
    Returns:
        Batched DataResult with padded sequences and attention mask in metadata
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # Extract prices and holdings lists
    prices_list = [item.prices for item in batch]
    holdings_list = [item.holdings for item in batch]
    
    # Check tensor shapes for debugging if needed
    # for i, (prices, holdings) in enumerate(zip(prices_list, holdings_list)):
    #     print(f"  Item {i}: prices.shape={prices.shape}, holdings.shape={holdings.shape}")
    
    # Handle different tensor shapes - squeeze batch dimension if present
    # Expected shape for pad_sequence: [seq_len, features]
    processed_prices = []
    processed_holdings = []
    
    for prices, holdings in zip(prices_list, holdings_list):
        if prices.dim() == 3 and prices.shape[0] == 1:
            # Shape: [1, seq_len, features] -> [seq_len, features]
            prices = prices.squeeze(0)
            holdings = holdings.squeeze(0)
        elif prices.dim() == 2:
            # Shape: [seq_len, features] - already correct
            pass
        else:
            raise ValueError(f"Unexpected tensor shape: {prices.shape}. Expected [seq_len, features] or [1, seq_len, features]")
        
        processed_prices.append(prices)
        processed_holdings.append(holdings)
    
    # Debug processed shapes if needed
    # for i, (prices, holdings) in enumerate(zip(processed_prices, processed_holdings)):
    #     print(f"  Item {i}: prices.shape={prices.shape}, holdings.shape={holdings.shape}")
    
    # Get original sequence lengths
    original_lengths = [tensor.shape[0] for tensor in processed_prices]
    
    # Pad sequences to same length
    # pad_sequence expects tensors with shape [seq_len, features]
    padded_prices = torch.nn.utils.rnn.pad_sequence(
        processed_prices, batch_first=True, padding_value=0.0
    )  # Shape: [batch_size, max_seq_len, features]
    
    padded_holdings = torch.nn.utils.rnn.pad_sequence(
        processed_holdings, batch_first=True, padding_value=0.0
    )  # Shape: [batch_size, max_seq_len, features]
    
    # Create attention mask
    attention_mask = create_attention_mask(padded_prices, original_lengths)
    
    # Handle time_features if present
    padded_time_features = None
    if batch[0].time_features is not None:
        time_features_list = [item.time_features for item in batch]
        padded_time_features = torch.nn.utils.rnn.pad_sequence(
            time_features_list, batch_first=True, padding_value=0.0
        )
    
    # Handle history_features if present
    padded_history_features = None
    if batch[0].history_features is not None:
        history_features_list = [item.history_features for item in batch]
        padded_history_features = torch.nn.utils.rnn.pad_sequence(
            history_features_list, batch_first=True, padding_value=0.0
        )
    
    # Handle delta_features if present
    padded_delta_features = None
    if batch[0].delta_features is not None:
        delta_features_list = [item.delta_features for item in batch]
        padded_delta_features = torch.nn.utils.rnn.pad_sequence(
            delta_features_list, batch_first=True, padding_value=0.0
        )
    
    # Merge metadata
    merged_metadata = {
        'batch_size': len(batch),
        'dataset_type': batch[0].metadata.get('dataset_type', 'unknown'),
        'is_variable_length': True,
        'original_lengths': original_lengths,
        'max_sequence_length': padded_prices.shape[1],
        'attention_mask': attention_mask
    }
    
    # Add common metadata keys
    for key in batch[0].metadata:
        if key not in ['dataset_type']:
            values = [item.metadata.get(key) for item in batch]
            # Only include if all items have the same value
            if all(v == values[0] for v in values):
                merged_metadata[key] = values[0]
            else:
                merged_metadata[f'{key}_list'] = values
    
    return DataResult(
        prices=padded_prices,
        holdings=padded_holdings,
        metadata=merged_metadata,
        time_features=padded_time_features,
        history_features=padded_history_features,
        delta_features=padded_delta_features
    )