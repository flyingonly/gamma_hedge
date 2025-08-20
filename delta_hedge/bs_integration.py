"""
Black-Scholes Integration Module
===============================

Integrates existing Black-Scholes and implied volatility tools for delta hedging.
Provides optimized batch calculations and enhanced functionality for options analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.black_scholes import BlackScholesModel
from tools.implied_volatility import ImpliedVolatilityCalculator
from .config import DeltaHedgeConfig

# Setup logging
logger = logging.getLogger(__name__)


class EnhancedBlackScholesCalculator:
    """
    Enhanced Black-Scholes calculator with batch processing and delta hedging optimizations.
    
    Extends the existing BlackScholesModel with:
    - Vectorized batch calculations
    - Integrated implied volatility calculation
    - Time series processing
    - Greeks calculation for portfolios
    """
    
    def __init__(self, config: DeltaHedgeConfig):
        """
        Initialize enhanced BS calculator.
        
        Args:
            config: Delta hedging configuration
        """
        self.config = config
        self.bs_model = BlackScholesModel()
        self.iv_calculator = ImpliedVolatilityCalculator(
            min_vol=config.black_scholes.min_volatility,
            max_vol=config.black_scholes.max_volatility,
            tolerance=config.black_scholes.calculation_tolerance
        )
        
        logger.info("Enhanced Black-Scholes calculator initialized")
    
    def calculate_option_price(self, S: Union[float, np.ndarray],
                             K: Union[float, np.ndarray],
                             T: Union[float, np.ndarray],
                             r: Union[float, np.ndarray],
                             sigma: Union[float, np.ndarray],
                             option_type: Union[str, List[str]]) -> np.ndarray:
        """
        Calculate option prices with support for batch processing.
        
        Args:
            S: Current underlying price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free rate(s)
            sigma: Volatility(ies)
            option_type: 'call', 'put', or list of types
            
        Returns:
            Array of option prices
        """
        # Convert inputs to arrays
        S = np.atleast_1d(S)
        K = np.atleast_1d(K)
        T = np.atleast_1d(T)
        r = np.atleast_1d(r)
        sigma = np.atleast_1d(sigma)
        
        # Handle option types
        if isinstance(option_type, str):
            option_types = [option_type] * len(S)
        else:
            option_types = option_type
        
        # Broadcast arrays to same shape
        max_len = max(len(S), len(K), len(T), len(r), len(sigma))
        S = np.broadcast_to(S, max_len) if len(S) < max_len else S
        K = np.broadcast_to(K, max_len) if len(K) < max_len else K
        T = np.broadcast_to(T, max_len) if len(T) < max_len else T
        r = np.broadcast_to(r, max_len) if len(r) < max_len else r
        sigma = np.broadcast_to(sigma, max_len) if len(sigma) < max_len else sigma
        
        # Calculate prices
        prices = np.zeros(max_len)
        for i in range(max_len):
            prices[i] = self.bs_model.option_price(
                S=float(S[i]), K=float(K[i]), T=float(T[i]),
                r=float(r[i]), sigma=float(sigma[i]), 
                option_type=option_types[i % len(option_types)]
            )
        
        return prices
    
    def calculate_greeks_batch(self, S: Union[float, np.ndarray],
                              K: Union[float, np.ndarray],
                              T: Union[float, np.ndarray],
                              r: Union[float, np.ndarray],
                              sigma: Union[float, np.ndarray],
                              option_type: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Calculate Greeks for batch of options.
        
        Args:
            S: Current underlying price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free rate(s)
            sigma: Volatility(ies)
            option_type: 'call', 'put', or list of types
            
        Returns:
            Dictionary with arrays for each Greek
        """
        # Convert inputs to arrays
        S = np.atleast_1d(S)
        K = np.atleast_1d(K)
        T = np.atleast_1d(T)
        r = np.atleast_1d(r)
        sigma = np.atleast_1d(sigma)
        
        # Handle option types
        if isinstance(option_type, str):
            option_types = [option_type] * len(S)
        else:
            option_types = option_type
        
        # Broadcast arrays to same shape
        max_len = max(len(S), len(K), len(T), len(r), len(sigma))
        S = np.broadcast_to(S, max_len) if len(S) < max_len else S
        K = np.broadcast_to(K, max_len) if len(K) < max_len else K
        T = np.broadcast_to(T, max_len) if len(T) < max_len else T
        r = np.broadcast_to(r, max_len) if len(r) < max_len else r
        sigma = np.broadcast_to(sigma, max_len) if len(sigma) < max_len else sigma
        
        # Initialize result arrays
        greeks = {
            'delta': np.zeros(max_len),
            'gamma': np.zeros(max_len),
            'theta': np.zeros(max_len),
            'vega': np.zeros(max_len),
            'rho': np.zeros(max_len)
        }
        
        # Calculate Greeks for each option
        for i in range(max_len):
            option_greeks = self.bs_model.calculate_greeks(
                S=float(S[i]), K=float(K[i]), T=float(T[i]),
                r=float(r[i]), sigma=float(sigma[i]),
                option_type=option_types[i % len(option_types)]
            )
            
            for greek_name, greek_value in option_greeks.items():
                greeks[greek_name][i] = greek_value
        
        return greeks
    
    def calculate_implied_volatility_batch(self, market_prices: np.ndarray,
                                         S: Union[float, np.ndarray],
                                         K: Union[float, np.ndarray],
                                         T: Union[float, np.ndarray],
                                         r: Union[float, np.ndarray],
                                         option_type: Union[str, List[str]],
                                         method: str = 'auto') -> np.ndarray:
        """
        Calculate implied volatilities for batch of options.
        
        Args:
            market_prices: Observed market prices
            S: Current underlying price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free rate(s)
            option_type: 'call', 'put', or list of types
            method: IV calculation method
            
        Returns:
            Array of implied volatilities
        """
        market_prices = np.atleast_1d(market_prices)
        S = np.atleast_1d(S)
        K = np.atleast_1d(K)
        T = np.atleast_1d(T)
        r = np.atleast_1d(r)
        
        # Handle option types
        if isinstance(option_type, str):
            option_types = [option_type] * len(market_prices)
        else:
            option_types = option_type
        
        # Broadcast arrays to same shape
        max_len = len(market_prices)
        S = np.broadcast_to(S, max_len) if len(S) < max_len else S
        K = np.broadcast_to(K, max_len) if len(K) < max_len else K
        T = np.broadcast_to(T, max_len) if len(T) < max_len else T
        r = np.broadcast_to(r, max_len) if len(r) < max_len else r
        
        # Calculate implied volatilities
        implied_vols = np.zeros(max_len)
        for i in range(max_len):
            try:
                iv = self.iv_calculator.calculate_iv(
                    market_price=float(market_prices[i]),
                    S=float(S[i]), K=float(K[i]), T=float(T[i]),
                    r=float(r[i]), option_type=option_types[i % len(option_types)],
                    method=method
                )
                implied_vols[i] = iv if not np.isnan(iv) else self._get_fallback_volatility(S[i], K[i])
            except Exception as e:
                logger.warning(f"IV calculation failed for option {i}: {e}")
                implied_vols[i] = self._get_fallback_volatility(S[i], K[i])
        
        return implied_vols
    
    def process_options_timeseries(self, options_data: Dict[str, Dict[str, np.ndarray]],
                                 underlying_prices: np.ndarray,
                                 timestamps: np.ndarray,
                                 contract_info: Dict) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process time series of options data to calculate prices and Greeks.
        
        Args:
            options_data: Dictionary of options data by option ID
            underlying_prices: Array of underlying prices over time
            timestamps: Array of timestamps
            contract_info: Contract specifications
            
        Returns:
            Dictionary with calculated option metrics over time
        """
        results = {}
        
        # Extract contract parameters
        expiry_date = pd.to_datetime(contract_info.get('expiry_date', '2025-12-31'))
        risk_free_rate = self.config.black_scholes.risk_free_rate
        
        for option_id, option_data in options_data.items():
            logger.info(f"Processing time series for {option_id}")
            
            # Extract option parameters
            strikes = option_data['strike']
            option_types = option_data['option_type']
            market_prices = option_data['last_price']
            
            # Calculate time to expiration for each timestamp
            timestamps_dt = pd.to_datetime(timestamps)
            times_to_expiry = np.array([
                max(1/365, (expiry_date - ts).total_seconds() / (365.25 * 24 * 3600))
                for ts in timestamps_dt
            ])
            
            # Calculate implied volatilities
            logger.debug(f"Calculating implied volatilities for {option_id}")
            implied_vols = self.calculate_implied_volatility_batch(
                market_prices=market_prices,
                S=underlying_prices,
                K=strikes,
                T=times_to_expiry,
                r=np.full_like(underlying_prices, risk_free_rate),
                option_type=option_types[0] if len(set(option_types)) == 1 else list(option_types)
            )
            
            # Validate and improve implied volatilities
            implied_vols = self._validate_and_improve_implied_vols(
                implied_vols, underlying_prices, strikes
            )
            
            # Calculate theoretical prices and Greeks
            logger.debug(f"Calculating Greeks for {option_id}")
            greeks = self.calculate_greeks_batch(
                S=underlying_prices,
                K=strikes,
                T=times_to_expiry,
                r=np.full_like(underlying_prices, risk_free_rate),
                sigma=implied_vols,
                option_type=option_types[0] if len(set(option_types)) == 1 else list(option_types)
            )
            
            # Calculate theoretical prices
            theoretical_prices = self.calculate_option_price(
                S=underlying_prices,
                K=strikes,
                T=times_to_expiry,
                r=np.full_like(underlying_prices, risk_free_rate),
                sigma=implied_vols,
                option_type=option_types[0] if len(set(option_types)) == 1 else list(option_types)
            )
            
            results[option_id] = {
                'market_prices': market_prices,
                'theoretical_prices': theoretical_prices,
                'implied_volatilities': implied_vols,
                'times_to_expiry': times_to_expiry,
                'strikes': strikes,
                'option_types': option_types,
                **greeks  # Add all Greeks to results
            }
        
        logger.info(f"Processed time series for {len(results)} options")
        return results
    
    def calculate_portfolio_greeks(self, options_results: Dict[str, Dict[str, np.ndarray]],
                                 position_sizes: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Calculate portfolio-level Greeks from individual option Greeks.
        
        Args:
            options_results: Results from process_options_timeseries
            position_sizes: Dictionary mapping option IDs to position sizes
            
        Returns:
            Dictionary with portfolio Greeks over time
        """
        if not options_results:
            return {}
        
        # Get time series length from first option
        first_option = next(iter(options_results.values()))
        n_timesteps = len(first_option['delta'])
        
        # Initialize portfolio Greeks
        portfolio_greeks = {
            'delta': np.zeros(n_timesteps),
            'gamma': np.zeros(n_timesteps),
            'theta': np.zeros(n_timesteps),
            'vega': np.zeros(n_timesteps),
            'rho': np.zeros(n_timesteps)
        }
        
        # Sum weighted Greeks across all positions
        for option_id, option_results in options_results.items():
            position_size = position_sizes.get(option_id, 0.0)
            
            if position_size != 0:
                for greek_name in portfolio_greeks.keys():
                    portfolio_greeks[greek_name] += (
                        position_size * option_results[greek_name]
                    )
        
        return portfolio_greeks
    
    def calculate_delta_hedging_ratio(self, portfolio_delta: np.ndarray,
                                    underlying_price: np.ndarray,
                                    hedge_method: str = "delta_neutral") -> np.ndarray:
        """
        Calculate hedging ratios for delta neutrality.
        
        Args:
            portfolio_delta: Portfolio delta over time
            underlying_price: Underlying asset price over time
            hedge_method: Hedging method ('delta_neutral', 'risk_adjusted')
            
        Returns:
            Array of hedge ratios (negative of portfolio delta for delta neutrality)
        """
        if hedge_method == "delta_neutral":
            # Simple delta neutrality: hedge ratio = -portfolio_delta
            hedge_ratios = -portfolio_delta
        elif hedge_method == "risk_adjusted":
            # Risk-adjusted hedging (could include transaction costs, etc.)
            # For now, just apply a dampening factor
            dampening_factor = 0.95
            hedge_ratios = -portfolio_delta * dampening_factor
        else:
            raise ValueError(f"Unknown hedge method: {hedge_method}")
        
        # Apply constraints
        max_ratio = self.config.delta_calculation.max_hedge_ratio
        hedge_ratios = np.clip(hedge_ratios, -max_ratio, max_ratio)
        
        return hedge_ratios
    
    def validate_calculations(self, results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, bool]:
        """
        Validate calculation results for reasonableness.
        
        Args:
            results: Calculation results to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        for option_id, option_results in results.items():
            option_validation = {}
            
            # Check for NaN or infinite values
            for metric_name, values in option_results.items():
                if isinstance(values, np.ndarray):
                    option_validation[f"{metric_name}_finite"] = np.isfinite(values).all()
                    option_validation[f"{metric_name}_not_nan"] = ~np.isnan(values).any()
            
            # Check reasonableness of specific metrics
            if 'delta' in option_results:
                deltas = option_results['delta']
                option_validation['delta_range_ok'] = np.all((-1.5 <= deltas) & (deltas <= 1.5))
            
            if 'implied_volatilities' in option_results:
                ivs = option_results['implied_volatilities']
                iv_min = self.config.black_scholes.min_volatility
                iv_max = self.config.black_scholes.max_volatility
                option_validation['iv_range_ok'] = np.all((iv_min <= ivs) & (ivs <= iv_max))
            
            if 'theoretical_prices' in option_results:
                prices = option_results['theoretical_prices']
                option_validation['prices_positive'] = np.all(prices >= 0)
            
            validation[option_id] = option_validation
        
        return validation
    
    def _get_fallback_volatility(self, underlying_price: float, strike_price: float) -> float:
        """
        Get improved fallback volatility based on asset characteristics.
        
        Args:
            underlying_price: Current underlying price
            strike_price: Strike price of the option
            
        Returns:
            Appropriate fallback volatility
        """
        # Determine asset type based on price level (crude heuristic)
        if underlying_price < 150:  # Likely bond futures (TU, FV, TY, US)
            base_vol = 0.12  # 12% for bond futures
        else:  # Likely equity index or commodity futures
            base_vol = 0.22  # 22% for equity/commodity futures
        
        # Adjust based on moneyness - OTM options typically have higher implied vol
        moneyness = underlying_price / strike_price
        if moneyness < 0.95 or moneyness > 1.05:  # OTM options
            vol_adjustment = 1.0 + 0.1 * abs(1.0 - moneyness)  # Up to 10% increase
            base_vol *= min(vol_adjustment, 1.5)  # Cap at 50% increase
        
        # Ensure within configured bounds
        return np.clip(base_vol, 
                      self.config.black_scholes.min_volatility,
                      self.config.black_scholes.max_volatility)
    
    def _validate_and_improve_implied_vols(self, implied_vols: np.ndarray, 
                                         underlying_prices: np.ndarray,
                                         strikes: np.ndarray) -> np.ndarray:
        """
        Validate and improve implied volatility estimates using cross-sectional consistency.
        
        Args:
            implied_vols: Initial implied volatilities
            underlying_prices: Underlying prices
            strikes: Strike prices
            
        Returns:
            Improved implied volatilities
        """
        improved_vols = implied_vols.copy()
        
        # Find obviously bad values and replace with interpolated or smoothed values
        valid_mask = (
            (implied_vols >= self.config.black_scholes.min_volatility) &
            (implied_vols <= self.config.black_scholes.max_volatility) &
            np.isfinite(implied_vols)
        )
        
        if np.sum(valid_mask) < len(implied_vols) * 0.5:  # Too many invalid values
            logger.warning("Many invalid implied volatilities, using fallback method")
            for i in range(len(implied_vols)):
                improved_vols[i] = self._get_fallback_volatility(underlying_prices[i], strikes[i])
        else:
            # Interpolate missing values
            if np.sum(~valid_mask) > 0:
                # Simple interpolation for invalid values
                median_vol = np.median(implied_vols[valid_mask])
                improved_vols[~valid_mask] = median_vol
        
        return improved_vols


def calculate_historical_volatility(prices: np.ndarray, window: int = 30) -> np.ndarray:
    """
    Calculate rolling historical volatility from price series.
    
    Args:
        prices: Array of prices
        window: Rolling window size in periods
        
    Returns:
        Array of historical volatilities (annualized)
    """
    if len(prices) < window + 1:
        return np.full(len(prices), 0.2)  # Default 20% volatility
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    # Calculate rolling volatility
    volatilities = np.zeros(len(prices))
    volatilities[:window] = np.std(returns[:window]) * np.sqrt(252)  # Annualized
    
    for i in range(window, len(prices)):
        vol_window = returns[i-window:i]
        volatilities[i] = np.std(vol_window) * np.sqrt(252)
    
    return volatilities


def estimate_time_to_expiry(current_date: datetime, expiry_date: datetime) -> float:
    """
    Estimate time to expiry in years.
    
    Args:
        current_date: Current date
        expiry_date: Option expiry date
        
    Returns:
        Time to expiry in years
    """
    time_diff = expiry_date - current_date
    days_to_expiry = time_diff.total_seconds() / (24 * 3600)
    return max(1/365, days_to_expiry / 365.25)  # Minimum 1 day


def interpolate_risk_free_rate(timestamps: np.ndarray, base_rate: float = 0.05) -> np.ndarray:
    """
    Generate risk-free rate time series (placeholder for more sophisticated implementation).
    
    Args:
        timestamps: Array of timestamps
        base_rate: Base risk-free rate
        
    Returns:
        Array of risk-free rates
    """
    # For now, return constant rate
    # In practice, this could interpolate from yield curve data
    return np.full(len(timestamps), base_rate)