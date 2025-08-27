"""
Historical Feature Calculator for Policy Network
==============================================

Calculates statistical historical features based on price and execution history
to provide market context H as required by optimal execution theory.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from utils.logger import get_logger
import logging

logger = get_logger(__name__)


@dataclass
class HistoricalFeatures:
    """Container for calculated historical features"""
    
    # Price history features (normalized 0-1)
    rolling_volatility_5: torch.Tensor    # 5-period rolling volatility
    rolling_volatility_10: torch.Tensor   # 10-period rolling volatility
    rolling_volatility_20: torch.Tensor   # 20-period rolling volatility
    price_momentum_short: torch.Tensor    # 3-period momentum
    price_momentum_long: torch.Tensor     # 10-period momentum
    price_ma_deviation: torch.Tensor      # deviation from moving average
    
    # Execution history features (normalized 0-1)
    execution_frequency_5: torch.Tensor   # execution rate in last 5 steps
    execution_frequency_10: torch.Tensor  # execution rate in last 10 steps
    steps_since_last_execution: torch.Tensor  # normalized steps since execution
    cumulative_execution_ratio: torch.Tensor  # total executed vs required
    
    # Position history features (normalized 0-1)
    position_deviation: torch.Tensor      # cumulative deviation from target
    position_trend: torch.Tensor          # position change trend
    unexecuted_delta: torch.Tensor        # remaining unexecuted delta
    position_variance: torch.Tensor       # position stability measure
    
    # Delta history features (NEW - for enhanced time sensitivity)
    delta_momentum_5: torch.Tensor        # 5-period delta momentum
    delta_volatility_5: torch.Tensor      # 5-period delta volatility
    delta_acceleration: torch.Tensor      # delta change acceleration
    
    def to_tensor(self) -> torch.Tensor:
        """Convert all features to single concatenated tensor"""
        features = [
            # Price features (6 features)
            self.rolling_volatility_5.unsqueeze(-1),
            self.rolling_volatility_10.unsqueeze(-1), 
            self.rolling_volatility_20.unsqueeze(-1),
            self.price_momentum_short.unsqueeze(-1),
            self.price_momentum_long.unsqueeze(-1),
            self.price_ma_deviation.unsqueeze(-1),
            
            # Execution features (4 features)
            self.execution_frequency_5.unsqueeze(-1),
            self.execution_frequency_10.unsqueeze(-1),
            self.steps_since_last_execution.unsqueeze(-1),
            self.cumulative_execution_ratio.unsqueeze(-1),
            
            # Position features (4 features)
            self.position_deviation.unsqueeze(-1),
            self.position_trend.unsqueeze(-1),
            self.unexecuted_delta.unsqueeze(-1),
            self.position_variance.unsqueeze(-1),
            
            # Delta features (3 features)
            self.delta_momentum_5.unsqueeze(-1),
            self.delta_volatility_5.unsqueeze(-1),
            self.delta_acceleration.unsqueeze(-1)
        ]
        
        # Concatenate all features: shape (seq_len, 17)
        return torch.cat(features, dim=-1)
    
    @property 
    def feature_dim(self) -> int:
        """Total number of historical features"""
        return 17  # 6 price + 4 execution + 4 position + 3 delta


class HistoricalFeatureCalculator:
    """
    Calculator for statistical historical features used by policy network.
    Provides market history context H as required by optimal execution theory.
    """
    
    def __init__(self, 
                 lookback_windows: Dict[str, int] = None,
                 normalization_method: str = 'minmax'):
        """
        Initialize historical feature calculator.
        
        Args:
            lookback_windows: Custom lookback periods for different features
            normalization_method: 'minmax', 'zscore', or 'robust'
        """
        # Default lookback windows
        self.windows = lookback_windows or {
            'volatility_short': 5,
            'volatility_medium': 10, 
            'volatility_long': 20,
            'momentum_short': 3,
            'momentum_long': 10,
            'execution_short': 5,
            'execution_medium': 10,
            'ma_period': 10
        }
        
        self.normalization_method = normalization_method
        
        # Feature statistics for normalization (updated during training)
        self.feature_stats = {}
        
    def calculate_features(self, 
                         prices: torch.Tensor,
                         holdings: torch.Tensor,
                         execution_history: Optional[torch.Tensor] = None,
                         delta_features: Optional[torch.Tensor] = None) -> HistoricalFeatures:
        """
        Calculate historical features for a sequence.
        
        Args:
            prices: Price sequence (seq_len, n_assets)
            holdings: Target holding sequence (seq_len, n_assets) 
            execution_history: Binary execution history (seq_len, n_assets), optional
            delta_features: Delta feature sequence (seq_len, 3) [delta, delta_change, delta_accel], optional
            
        Returns:
            HistoricalFeatures object containing all computed features
        """
        seq_len = prices.shape[0]
        device = prices.device
        dtype = prices.dtype
        
        # If no execution history provided, assume uniform execution
        if execution_history is None:
            execution_history = torch.ones_like(prices)
        
        # Calculate price-based features
        price_features = self._calculate_price_features(prices)
        
        # Calculate execution-based features
        execution_features = self._calculate_execution_features(execution_history)
        
        # Calculate position-based features
        position_features = self._calculate_position_features(holdings, execution_history)
        
        # Calculate delta-based features (NEW)
        delta_hist_features = self._calculate_delta_features(delta_features)
        
        # Combine into HistoricalFeatures object
        return HistoricalFeatures(
            # Price features
            rolling_volatility_5=price_features['volatility_5'],
            rolling_volatility_10=price_features['volatility_10'],
            rolling_volatility_20=price_features['volatility_20'],
            price_momentum_short=price_features['momentum_short'],
            price_momentum_long=price_features['momentum_long'],
            price_ma_deviation=price_features['ma_deviation'],
            
            # Execution features
            execution_frequency_5=execution_features['freq_5'],
            execution_frequency_10=execution_features['freq_10'],
            steps_since_last_execution=execution_features['steps_since_last'],
            cumulative_execution_ratio=execution_features['cumulative_ratio'],
            
            # Position features
            position_deviation=position_features['deviation'],
            position_trend=position_features['trend'],
            unexecuted_delta=position_features['unexecuted'],
            position_variance=position_features['variance'],
            
            # Delta features (NEW)
            delta_momentum_5=delta_hist_features['momentum_5'],
            delta_volatility_5=delta_hist_features['volatility_5'],
            delta_acceleration=delta_hist_features['acceleration']
        )
    
    def _calculate_price_features(self, prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate price-based historical features"""
        seq_len = prices.shape[0]
        
        # Use first asset for price features (assuming single underlying)
        price_series = prices[:, 0] if prices.shape[1] > 1 else prices.squeeze(-1)
        
        # Calculate returns for volatility computation
        returns = torch.diff(torch.log(price_series + 1e-8), dim=0)
        returns = torch.cat([torch.zeros(1, device=returns.device), returns])
        
        features = {}
        
        # Rolling volatilities
        for window_name, window_size in [
            ('volatility_5', self.windows['volatility_short']),
            ('volatility_10', self.windows['volatility_medium']),
            ('volatility_20', self.windows['volatility_long'])
        ]:
            volatilities = []
            for i in range(seq_len):
                start_idx = max(0, i - window_size + 1)
                window_returns = returns[start_idx:i+1]
                vol = torch.std(window_returns) if len(window_returns) > 1 else torch.tensor(0.0, device=returns.device)
                volatilities.append(vol)
            
            features[window_name] = self._normalize_feature(torch.stack(volatilities), window_name)
        
        # Price momentum (rate of change)
        for momentum_name, window_size in [
            ('momentum_short', self.windows['momentum_short']),
            ('momentum_long', self.windows['momentum_long'])
        ]:
            momentums = []
            for i in range(seq_len):
                if i >= window_size:
                    momentum = (price_series[i] - price_series[i - window_size]) / price_series[i - window_size]
                else:
                    momentum = torch.tensor(0.0, device=price_series.device)
                momentums.append(momentum)
            
            features[momentum_name] = self._normalize_feature(torch.stack(momentums), momentum_name)
        
        # Moving average deviation
        ma_period = self.windows['ma_period']
        ma_deviations = []
        for i in range(seq_len):
            start_idx = max(0, i - ma_period + 1)
            window_prices = price_series[start_idx:i+1]
            ma = torch.mean(window_prices)
            deviation = (price_series[i] - ma) / ma if ma > 0 else torch.tensor(0.0, device=price_series.device)
            ma_deviations.append(deviation)
        
        features['ma_deviation'] = self._normalize_feature(torch.stack(ma_deviations), 'ma_deviation')
        
        return features
    
    def _calculate_execution_features(self, execution_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate execution-based historical features"""
        seq_len = execution_history.shape[0]
        
        # Use first asset for execution features
        exec_series = execution_history[:, 0] if execution_history.shape[1] > 1 else execution_history.squeeze(-1)
        
        features = {}
        
        # Execution frequencies
        for freq_name, window_size in [
            ('freq_5', self.windows['execution_short']),
            ('freq_10', self.windows['execution_medium'])
        ]:
            frequencies = []
            for i in range(seq_len):
                start_idx = max(0, i - window_size + 1)
                window_executions = exec_series[start_idx:i+1]
                freq = torch.mean(window_executions.float())
                frequencies.append(freq)
            
            features[freq_name] = torch.stack(frequencies)  # Already normalized (0-1)
        
        # Steps since last execution
        steps_since = []
        last_execution_idx = -1
        
        for i in range(seq_len):
            if exec_series[i] > 0.5:  # Execution occurred
                last_execution_idx = i
                steps = 0
            else:
                steps = i - last_execution_idx if last_execution_idx >= 0 else i + 1
            
            steps_since.append(steps)
        
        # Normalize steps by sequence length
        max_steps = max(steps_since) if steps_since else 1
        max_steps = max(max_steps, 1)  # Ensure no division by zero
        features['steps_since_last'] = torch.tensor(
            [s / max_steps for s in steps_since], 
            device=exec_series.device, 
            dtype=exec_series.dtype
        )
        
        # Cumulative execution ratio
        cumulative_ratios = []
        total_possible = torch.arange(1, seq_len + 1, device=exec_series.device, dtype=torch.float32)
        cumulative_executions = torch.cumsum(exec_series, dim=0)
        
        features['cumulative_ratio'] = cumulative_executions / total_possible
        
        return features
    
    def _calculate_position_features(self, 
                                   holdings: torch.Tensor,
                                   execution_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate position-based historical features"""
        seq_len = holdings.shape[0]
        
        # Use first asset for position features
        holding_series = holdings[:, 0] if holdings.shape[1] > 1 else holdings.squeeze(-1)
        exec_series = execution_history[:, 0] if execution_history.shape[1] > 1 else execution_history.squeeze(-1)
        
        features = {}
        
        # Position deviation (cumulative difference from target)
        deviations = []
        cumulative_deviation = 0.0
        
        for i in range(seq_len):
            target_change = holding_series[i] - (holding_series[i-1] if i > 0 else 0)
            executed_change = target_change * exec_series[i]
            deviation = target_change - executed_change
            cumulative_deviation += deviation.item()
            deviations.append(abs(cumulative_deviation))
        
        max_deviation = max(deviations) if deviations else 1.0
        max_deviation = max(max_deviation, 1e-8)  # Prevent division by zero
        features['deviation'] = self._normalize_feature(
            torch.tensor(deviations, device=holdings.device, dtype=holdings.dtype) / max_deviation,
            'deviation'
        )
        
        # Position trend (moving average of position changes)  
        trend_window = 5
        trends = []
        for i in range(seq_len):
            if i == 0:
                trend = torch.tensor(0.0, device=holdings.device)
            else:
                start_idx = max(0, i - trend_window + 1)
                end_idx = i + 1
                
                # Calculate changes within the window
                window_positions = holding_series[start_idx:end_idx]
                if len(window_positions) > 1:
                    # Calculate mean of changes within window
                    window_changes = torch.diff(window_positions)
                    trend = torch.mean(window_changes) if len(window_changes) > 0 else torch.tensor(0.0, device=holdings.device)
                else:
                    trend = torch.tensor(0.0, device=holdings.device)
            trends.append(trend)
        
        features['trend'] = self._normalize_feature(torch.stack(trends), 'trend')
        
        # Unexecuted delta (remaining execution needed)
        unexecuted = []
        total_unexecuted = 0.0
        
        for i in range(seq_len):
            required_change = holding_series[i] - (holding_series[i-1] if i > 0 else 0)
            executed_amount = required_change * exec_series[i]
            unexecuted_amount = required_change - executed_amount
            total_unexecuted += unexecuted_amount.item()
            unexecuted.append(abs(total_unexecuted))
        
        max_unexecuted = max(unexecuted) if unexecuted else 1.0
        max_unexecuted = max(max_unexecuted, 1e-8)  # Prevent division by zero
        features['unexecuted'] = torch.tensor(
            [u / max_unexecuted for u in unexecuted],
            device=holdings.device,
            dtype=holdings.dtype
        )
        
        # Position variance (stability measure)
        variance_window = 10
        variances = []
        for i in range(seq_len):
            start_idx = max(0, i - variance_window + 1)
            window_positions = holding_series[start_idx:i+1]
            var = torch.var(window_positions) if len(window_positions) > 1 else torch.tensor(0.0, device=holdings.device)
            variances.append(var)
        
        features['variance'] = self._normalize_feature(torch.stack(variances), 'variance')
        
        return features
    
    def _normalize_feature(self, feature: torch.Tensor, feature_name: str) -> torch.Tensor:
        """Normalize feature values to [0, 1] range"""
        if self.normalization_method == 'minmax':
            min_val = torch.min(feature)
            max_val = torch.max(feature)
            if max_val > min_val:
                normalized = (feature - min_val) / (max_val - min_val)
            else:
                normalized = torch.zeros_like(feature)
        
        elif self.normalization_method == 'zscore':
            mean_val = torch.mean(feature)
            std_val = torch.std(feature)
            if std_val > 0:
                normalized = torch.sigmoid((feature - mean_val) / std_val)
            else:
                normalized = torch.full_like(feature, 0.5)
        
        elif self.normalization_method == 'robust':
            median_val = torch.median(feature)
            mad = torch.median(torch.abs(feature - median_val))
            if mad > 0:
                normalized = torch.sigmoid((feature - median_val) / (1.4826 * mad))
            else:
                normalized = torch.full_like(feature, 0.5)
        
        else:
            # No normalization
            normalized = feature
        
        return normalized.clamp(0, 1)  # Ensure [0, 1] range
    
    def _calculate_delta_features(self, delta_features: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate delta-based historical features for enhanced time sensitivity"""
        
        # Default zero features if delta_features not provided
        if delta_features is None:
            seq_len = 100  # Default fallback
            device = torch.device('cpu')
        else:
            seq_len = delta_features.shape[0]
            device = delta_features.device
        
        features = {}
        
        if delta_features is not None:
            # Extract delta components: [current_delta, delta_change, delta_accel]
            current_delta = delta_features[:, 0]  # Current normalized delta
            delta_change = delta_features[:, 1]   # Delta change rate
            delta_accel = delta_features[:, 2]    # Delta acceleration
            
            # Feature 1: 5-period delta momentum
            window_size = min(5, seq_len)
            delta_momentum = torch.zeros_like(current_delta)
            for i in range(window_size, seq_len):
                recent_deltas = current_delta[i-window_size:i]
                # Calculate momentum as trend slope
                if window_size > 1:
                    x = torch.arange(window_size, dtype=torch.float, device=device)
                    y = recent_deltas
                    # Simple linear regression slope
                    n = window_size
                    sum_x = x.sum()
                    sum_y = y.sum()
                    sum_xy = (x * y).sum()
                    sum_x2 = (x * x).sum()
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
                    delta_momentum[i] = slope
            features['momentum_5'] = self._normalize_feature(delta_momentum, 'delta_momentum_5')
            
            # Feature 2: 5-period delta volatility
            delta_volatility = torch.zeros_like(current_delta)
            for i in range(window_size, seq_len):
                recent_changes = delta_change[i-window_size:i]
                volatility = torch.std(recent_changes, correction=0)
                delta_volatility[i] = volatility
            features['volatility_5'] = self._normalize_feature(delta_volatility, 'delta_volatility_5')
            
            # Feature 3: Delta acceleration (already computed in delta_features)
            features['acceleration'] = self._normalize_feature(delta_accel, 'delta_acceleration')
            
        else:
            # Fallback: zero features
            zero_feature = torch.zeros(seq_len, device=device)
            features = {
                'momentum_5': zero_feature,
                'volatility_5': zero_feature,
                'acceleration': zero_feature
            }
        
        return features
    
    @property
    def feature_dim(self) -> int:
        """Total number of historical features"""
        return 17  # Updated count: 6 price + 4 execution + 4 position + 3 delta