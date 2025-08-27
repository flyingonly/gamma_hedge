#!/usr/bin/env python3
"""
Centralized dimension management system for the Gamma Hedge project.

This module provides a unified approach to calculating and validating tensor dimensions
across the entire system, eliminating hardcoded values and preventing shape mismatch errors.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DimensionSpec:
    """
    Specification for model input dimensions with detailed breakdown.
    
    This dataclass provides a complete description of all dimension components
    and validates their consistency.
    """
    # Core features (always present)
    prev_holding: int = 1        # Previous holdings W(t_{i-1})
    current_holding: int = 1     # Current holdings W(t_i)  
    price: int = 1               # Asset prices A(t_i)
    
    # Optional features (may be present based on configuration)
    time_features: int = 0       # Time-based features (e.g., time_remaining)
    delta_features: int = 0      # Delta-related features [delta, delta_change, delta_accel]
    history_features: int = 0    # Statistical historical features
    
    # Computed properties
    base_features: int = field(init=False)  # Core features total
    total: int = field(init=False)          # All features total
    
    def __post_init__(self):
        """Calculate derived dimensions and validate consistency."""
        self.base_features = self.prev_holding + self.current_holding + self.price
        self.total = (self.base_features + self.time_features + 
                     self.delta_features + self.history_features)
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate dimension specification consistency.
        
        Returns:
            bool: True if dimensions are consistent
            
        Raises:
            ValueError: If dimensions are invalid or inconsistent
        """
        if self.total <= 0:
            raise ValueError(f"Total dimensions must be positive, got {self.total}")
        
        if self.base_features < 3:
            raise ValueError(f"Base features must be at least 3 (prev_holding + current_holding + price), got {self.base_features}")
        
        # Verify computed total matches sum of components
        expected_total = (self.prev_holding + self.current_holding + self.price + 
                         self.time_features + self.delta_features + self.history_features)
        if self.total != expected_total:
            raise ValueError(f"Dimension total mismatch: computed={self.total}, expected={expected_total}")
        
        return True
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for logging and debugging."""
        return {
            'prev_holding': self.prev_holding,
            'current_holding': self.current_holding,
            'price': self.price,
            'time_features': self.time_features,
            'delta_features': self.delta_features,
            'history_features': self.history_features,
            'base_features': self.base_features,
            'total': self.total
        }
    
    def __str__(self) -> str:
        """Human-readable dimension breakdown."""
        return (f"DimensionSpec(total={self.total}: "
                f"base={self.base_features}, time={self.time_features}, "
                f"delta={self.delta_features}, history={self.history_features})")


class DimensionCalculator:
    """
    Centralized dimension calculation and validation system.
    
    This class provides methods to:
    - Calculate model input dimensions dynamically
    - Auto-detect dimensions from data samples
    - Validate tensor shapes against expected dimensions
    - Provide clear debugging information for dimension mismatches
    """
    
    def __init__(self, n_assets: int = 1, enable_debugging: bool = True):
        """
        Initialize dimension calculator.
        
        Args:
            n_assets: Number of assets in the system (default: 1)
            enable_debugging: Enable detailed dimension debugging logs
        """
        self.n_assets = n_assets
        self.enable_debugging = enable_debugging
        self._cached_specs: Dict[str, DimensionSpec] = {}
        
        if self.enable_debugging:
            logger.info(f"DimensionCalculator initialized for {n_assets} asset(s)")
    
    def calculate_model_input_dim(self, 
                                 include_history: bool = True,
                                 include_time: bool = True, 
                                 include_delta: bool = True) -> DimensionSpec:
        """
        Calculate total input dimensions for model networks.
        
        Args:
            include_history: Include statistical historical features
            include_time: Include time-based features
            include_delta: Include delta-related features
            
        Returns:
            DimensionSpec: Complete dimension specification
        """
        # Create cache key for this configuration
        cache_key = f"model_input_{include_history}_{include_time}_{include_delta}_{self.n_assets}"
        
        if cache_key in self._cached_specs:
            if self.enable_debugging:
                logger.debug(f"Using cached dimension spec: {cache_key}")
            return self._cached_specs[cache_key]
        
        # Calculate dimensions
        spec = DimensionSpec(
            prev_holding=self.n_assets,
            current_holding=self.n_assets,
            price=self.n_assets,
            time_features=1 if include_time else 0,
            delta_features=3 if include_delta else 0,  # [current_delta, delta_change, delta_accel]
            history_features=self._get_history_feature_dim() if include_history else 0
        )
        
        # Cache the result
        self._cached_specs[cache_key] = spec
        
        if self.enable_debugging:
            logger.info(f"Calculated model input dimensions: {spec}")
            logger.debug(f"Dimension breakdown: {spec.to_dict()}")
        
        return spec
    
    def detect_feature_dimensions(self, sample_data) -> DimensionSpec:
        """
        Auto-detect dimensions from actual data sample.
        
        Args:
            sample_data: Data sample (DataResult or similar)
            
        Returns:
            DimensionSpec: Detected dimension specification
        """
        try:
            # Extract tensor shapes from sample data
            prices_dim = self._get_tensor_last_dim(sample_data.prices)
            holdings_dim = self._get_tensor_last_dim(sample_data.holdings)
            time_dim = self._get_tensor_last_dim(getattr(sample_data, 'time_features', None))
            delta_dim = self._get_tensor_last_dim(getattr(sample_data, 'delta_features', None))
            history_dim = self._get_tensor_last_dim(getattr(sample_data, 'history_features', None))
            
            spec = DimensionSpec(
                prev_holding=holdings_dim,
                current_holding=holdings_dim,
                price=prices_dim,
                time_features=time_dim,
                delta_features=delta_dim,
                history_features=history_dim
            )
            
            if self.enable_debugging:
                logger.info(f"Auto-detected dimensions from data: {spec}")
                logger.debug(f"Raw tensor shapes: prices={prices_dim}, holdings={holdings_dim}, "
                           f"time={time_dim}, delta={delta_dim}, history={history_dim}")
            
            return spec
            
        except Exception as e:
            logger.error(f"Failed to auto-detect dimensions: {e}")
            raise ValueError(f"Cannot auto-detect dimensions from sample data: {e}")
    
    def validate_tensor_shapes(self, 
                              tensors: Dict[str, torch.Tensor], 
                              expected_spec: DimensionSpec) -> bool:
        """
        Validate tensor shapes match expected dimensions.
        
        Args:
            tensors: Dictionary of tensor name -> tensor
            expected_spec: Expected dimension specification
            
        Returns:
            bool: True if all shapes are valid
            
        Raises:
            ValueError: If shapes don't match expected dimensions
        """
        errors = []
        
        for name, tensor in tensors.items():
            if tensor is None:
                continue
                
            expected_dim = getattr(expected_spec, name, None)
            if expected_dim is None:
                continue
                
            actual_dim = tensor.shape[-1] if len(tensor.shape) > 0 else 0
            
            if actual_dim != expected_dim:
                errors.append(f"{name}: expected {expected_dim}, got {actual_dim} (shape: {tensor.shape})")
        
        if errors:
            error_msg = f"Tensor shape validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            error_msg += f"\nExpected spec: {expected_spec}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.enable_debugging:
            logger.debug(f"Tensor shape validation passed for spec: {expected_spec}")
        
        return True
    
    def create_debug_summary(self, 
                           model_input_tensor: torch.Tensor,
                           expected_spec: DimensionSpec) -> str:
        """
        Create detailed debug summary for dimension mismatches.
        
        Args:
            model_input_tensor: Actual model input tensor
            expected_spec: Expected dimension specification
            
        Returns:
            str: Detailed debug information
        """
        actual_shape = model_input_tensor.shape
        actual_dim = actual_shape[-1] if len(actual_shape) > 0 else 0
        
        summary = [
            "=== DIMENSION DEBUG SUMMARY ===",
            f"Model input tensor shape: {actual_shape}",
            f"Actual input dimensions: {actual_dim}",
            f"Expected dimensions: {expected_spec.total}",
            "",
            "Expected dimension breakdown:",
        ]
        
        breakdown = expected_spec.to_dict()
        for feature_name, dim_count in breakdown.items():
            if dim_count > 0:
                summary.append(f"  {feature_name}: {dim_count}")
        
        if actual_dim != expected_spec.total:
            summary.extend([
                "",
                f"❌ MISMATCH: {actual_dim - expected_spec.total:+d} dimensions",
                "Possible causes:",
                "  - Missing features in model input concatenation",
                "  - Incorrect feature dimension calculations", 
                "  - Data preprocessing inconsistencies",
                "  - Model architecture mismatch"
            ])
        else:
            summary.append("\n✅ Dimensions match expected specification")
        
        return "\n".join(summary)
    
    def _get_history_feature_dim(self) -> int:
        """
        Get historical feature dimensions by inspecting the actual implementation.
        
        Returns:
            int: Number of historical features
        """
        try:
            from utils.historical_features import HistoricalFeatures, HistoricalFeatureCalculator
            
            if self.enable_debugging:
                logger.debug("Attempting to detect historical feature dimensions")
            
            # Use the feature_dim property from HistoricalFeatures class
            # Create a dummy instance with zero tensors to get the dimension count
            seq_len = 10
            dummy_features = HistoricalFeatures(
                rolling_volatility_5=torch.zeros(seq_len),
                rolling_volatility_10=torch.zeros(seq_len),
                rolling_volatility_20=torch.zeros(seq_len),
                price_momentum_short=torch.zeros(seq_len),
                price_momentum_long=torch.zeros(seq_len),
                price_ma_deviation=torch.zeros(seq_len),
                execution_frequency_5=torch.zeros(seq_len),
                execution_frequency_10=torch.zeros(seq_len),
                steps_since_last_execution=torch.zeros(seq_len),
                cumulative_execution_ratio=torch.zeros(seq_len),
                position_deviation=torch.zeros(seq_len),
                position_trend=torch.zeros(seq_len),
                unexecuted_delta=torch.zeros(seq_len),
                position_variance=torch.zeros(seq_len),
                delta_momentum_5=torch.zeros(seq_len),
                delta_volatility_5=torch.zeros(seq_len),
                delta_acceleration=torch.zeros(seq_len)
            )
            
            # Get dimension from to_tensor method
            features_tensor = dummy_features.to_tensor()
            result = features_tensor.shape[-1]
            
            if self.enable_debugging:
                logger.debug(f"Got historical feature dimension from tensor shape: {result}")
            return result
            
        except ImportError:
            if self.enable_debugging:
                logger.warning("Historical features not available, using default dimension")
            return 0
        except Exception as e:
            if self.enable_debugging:
                logger.warning(f"Failed to detect historical feature dimensions: {e}")
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
            return 0
    
    def _get_tensor_last_dim(self, tensor) -> int:
        """Get last dimension of tensor, or 0 if tensor is None/empty."""
        if tensor is None:
            return 0
        if isinstance(tensor, torch.Tensor):
            return tensor.shape[-1] if len(tensor.shape) > 0 else 0
        return 0
    
    def clear_cache(self):
        """Clear cached dimension specifications."""
        self._cached_specs.clear()
        if self.enable_debugging:
            logger.debug("Dimension cache cleared")


# Global instance for convenience
_global_calculator = None

def get_dimension_calculator(n_assets: int = 1, enable_debugging: bool = True) -> DimensionCalculator:
    """
    Get global dimension calculator instance.
    
    Args:
        n_assets: Number of assets
        enable_debugging: Enable debugging logs
        
    Returns:
        DimensionCalculator: Global calculator instance
    """
    global _global_calculator
    
    if _global_calculator is None or _global_calculator.n_assets != n_assets:
        _global_calculator = DimensionCalculator(n_assets, enable_debugging)
    
    return _global_calculator


def calculate_model_input_dim(n_assets: int = 1, 
                            include_history: bool = True,
                            include_time: bool = True,
                            include_delta: bool = True) -> int:
    """
    Convenience function for calculating model input dimensions.
    
    Args:
        n_assets: Number of assets
        include_history: Include historical features
        include_time: Include time features
        include_delta: Include delta features
        
    Returns:
        int: Total input dimensions
    """
    calculator = get_dimension_calculator(n_assets)
    spec = calculator.calculate_model_input_dim(include_history, include_time, include_delta)
    return spec.total


def validate_model_input(model_input: torch.Tensor, 
                        n_assets: int = 1,
                        include_history: bool = True,
                        include_time: bool = True, 
                        include_delta: bool = True) -> bool:
    """
    Convenience function for validating model input tensor dimensions.
    
    Args:
        model_input: Input tensor to validate
        n_assets: Number of assets
        include_history: Whether historical features are included
        include_time: Whether time features are included
        include_delta: Whether delta features are included
        
    Returns:
        bool: True if dimensions are valid
        
    Raises:
        ValueError: If dimensions don't match
    """
    calculator = get_dimension_calculator(n_assets)
    expected_spec = calculator.calculate_model_input_dim(include_history, include_time, include_delta)
    
    actual_dim = model_input.shape[-1] if len(model_input.shape) > 0 else 0
    
    if actual_dim != expected_spec.total:
        debug_summary = calculator.create_debug_summary(model_input, expected_spec)
        raise ValueError(f"Model input dimension mismatch:\n{debug_summary}")
    
    return True