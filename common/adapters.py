"""
Adapters for backward compatibility and data format conversion
"""

from typing import Union, Tuple, Any
import torch
from .interfaces import DataResult
from .exceptions import ValidationError

class LegacyDataAdapter:
    """Adapter for legacy data formats"""
    
    @staticmethod
    def from_tuple(data: Tuple[torch.Tensor, torch.Tensor]) -> DataResult:
        """Convert from tuple format (prices, holdings)"""
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValidationError(f"Expected tuple of length 2, got {type(data)} of length {len(data) if hasattr(data, '__len__') else 'unknown'}")
        
        prices, holdings = data
        if not isinstance(prices, torch.Tensor) or not isinstance(holdings, torch.Tensor):
            raise ValidationError(f"Expected torch.Tensor, got prices: {type(prices)}, holdings: {type(holdings)}")
        
        return DataResult(prices, holdings)
    
    @staticmethod
    def to_tuple(data: DataResult) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert DataResult to tuple format"""
        return data.extract_tensors()
    
    @staticmethod
    def from_market_data(market_data: Any) -> DataResult:
        """Convert from MarketData object (legacy)"""
        if not hasattr(market_data, 'prices') or not hasattr(market_data, 'holdings'):
            raise ValidationError(f"Object must have 'prices' and 'holdings' attributes")
        
        metadata = getattr(market_data, 'metadata', {})
        return DataResult(market_data.prices, market_data.holdings, metadata)

class DataFormatEnsurer:
    """Utility to ensure consistent data format"""
    
    @staticmethod
    def ensure_data_result(data: Union[DataResult, Tuple, Any]) -> DataResult:
        """Ensure data is in DataResult format"""
        if isinstance(data, DataResult):
            return data
        elif isinstance(data, tuple):
            return LegacyDataAdapter.from_tuple(data)
        elif hasattr(data, 'prices') and hasattr(data, 'holdings'):
            return LegacyDataAdapter.from_market_data(data)
        else:
            raise ValidationError(f"Cannot convert {type(data)} to DataResult")
    
    @staticmethod
    def extract_compatible_format(data: Union[DataResult, Tuple, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract tuple format for backward compatibility"""
        data_result = DataFormatEnsurer.ensure_data_result(data)
        return data_result.extract_tensors()

class BatchDataProcessor:
    """Process batch data with format flexibility"""
    
    @staticmethod
    def process_batch(batch_data: Union[DataResult, Tuple, list]) -> DataResult:
        """Process batch data into consistent format"""
        if isinstance(batch_data, list):
            # Handle list of DataResult objects
            if all(isinstance(item, DataResult) for item in batch_data):
                # Stack batch data
                prices_batch = torch.stack([item.prices for item in batch_data])
                holdings_batch = torch.stack([item.holdings for item in batch_data])
                
                # Merge metadata (take from first item, add batch info)
                metadata = batch_data[0].metadata.copy() if batch_data else {}
                metadata['batch_size'] = len(batch_data)
                
                return DataResult(prices_batch, holdings_batch, metadata)
            else:
                raise ValidationError("All items in batch list must be DataResult objects")
        
        return DataFormatEnsurer.ensure_data_result(batch_data)

# Utility functions for common conversions
def to_data_result(data: Any) -> DataResult:
    """Convenience function for data conversion"""
    return DataFormatEnsurer.ensure_data_result(data)

def to_tensor_tuple(data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience function for tuple extraction"""  
    return DataFormatEnsurer.extract_compatible_format(data)