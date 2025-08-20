"""
Implied Volatility Calculation Module
====================================

This module provides various methods for calculating implied volatility from market prices.
Implements Newton-Raphson, Brent's method, and other numerical approaches for IV calculation.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from .black_scholes import BlackScholesModel
import warnings
warnings.filterwarnings('ignore')


class ImpliedVolatilityCalculator:
    """
    Implied volatility calculator using multiple numerical methods.
    
    Supports:
    - Newton-Raphson method (fast convergence)
    - Brent's method (robust bracketing)
    - Bisection method (backup option)
    """
    
    def __init__(self, min_vol=0.001, max_vol=5.0, tolerance=1e-6, max_iterations=100):
        """
        Initialize implied volatility calculator.
        
        Parameters:
        -----------
        min_vol : float
            Minimum volatility bound (default 0.1%)
        max_vol : float
            Maximum volatility bound (default 500%)
        tolerance : float
            Numerical tolerance for convergence
        max_iterations : int
            Maximum iterations for iterative methods
        """
        self.min_vol = min_vol
        self.max_vol = max_vol
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.bs_model = BlackScholesModel()
    
    def newton_raphson_iv(self, market_price, S, K, T, r, option_type='call', initial_guess=0.2):
        """
        Calculate implied volatility using Newton-Raphson method.
        
        This method uses the first derivative (vega) to iteratively converge
        to the implied volatility that matches the market price.
        
        Parameters:
        -----------
        market_price : float
            Observed market price of the option
        S : float
            Current underlying price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        initial_guess : float
            Starting volatility guess
            
        Returns:
        --------
        float
            Implied volatility or np.nan if convergence fails
        """
        if T <= 0:
            return np.nan
        
        # Check for arbitrage bounds
        if not self._check_arbitrage_bounds(market_price, S, K, T, r, option_type):
            return np.nan
        
        sigma = initial_guess
        
        for i in range(self.max_iterations):
            try:
                # Calculate theoretical price and vega
                theoretical_price = self.bs_model.option_price(S, K, T, r, sigma, option_type)
                vega = self.bs_model.implied_volatility_vega(S, K, T, r, sigma)
                
                # Check if vega is too small (numerical instability)
                if abs(vega) < 1e-10:
                    break
                
                # Newton-Raphson update
                price_diff = theoretical_price - market_price
                sigma_new = sigma - price_diff / vega
                
                # Ensure sigma stays within bounds
                sigma_new = max(self.min_vol, min(self.max_vol, sigma_new))
                
                # Check for convergence
                if abs(sigma_new - sigma) < self.tolerance:
                    return sigma_new
                
                sigma = sigma_new
                
            except (OverflowError, ZeroDivisionError):
                break
        
        return sigma if self.min_vol <= sigma <= self.max_vol else np.nan
    
    def brent_method_iv(self, market_price, S, K, T, r, option_type='call'):
        """
        Calculate implied volatility using Brent's method.
        
        This is a bracketing method that's more robust but potentially slower
        than Newton-Raphson. Good for cases where Newton-Raphson fails.
        
        Returns:
        --------
        float
            Implied volatility or np.nan if no solution exists
        """
        if T <= 0:
            return np.nan
        
        if not self._check_arbitrage_bounds(market_price, S, K, T, r, option_type):
            return np.nan
        
        def objective_function(sigma):
            """Objective function: theoretical_price - market_price"""
            try:
                return self.bs_model.option_price(S, K, T, r, sigma, option_type) - market_price
            except:
                return float('inf')
        
        try:
            # Check if solution exists within bounds
            f_min = objective_function(self.min_vol)
            f_max = objective_function(self.max_vol)
            
            # Brent's method requires opposite signs at endpoints
            if f_min * f_max > 0:
                return np.nan
            
            # Apply Brent's method
            result = brentq(objective_function, self.min_vol, self.max_vol, xtol=self.tolerance)
            return result
            
        except (ValueError, RuntimeError):
            return np.nan
    
    def bisection_iv(self, market_price, S, K, T, r, option_type='call'):
        """
        Calculate implied volatility using bisection method.
        
        Simple and robust bracketing method. Slower convergence but very stable.
        Used as a fallback when other methods fail.
        """
        if T <= 0:
            return np.nan
        
        if not self._check_arbitrage_bounds(market_price, S, K, T, r, option_type):
            return np.nan
        
        low = self.min_vol
        high = self.max_vol
        
        # Check if solution exists
        f_low = self.bs_model.option_price(S, K, T, r, low, option_type) - market_price
        f_high = self.bs_model.option_price(S, K, T, r, high, option_type) - market_price
        
        if f_low * f_high > 0:
            return np.nan
        
        # Bisection iterations
        for _ in range(self.max_iterations):
            mid = (low + high) / 2
            f_mid = self.bs_model.option_price(S, K, T, r, mid, option_type) - market_price
            
            if abs(f_mid) < self.tolerance:
                return mid
            
            if f_low * f_mid < 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        
        return (low + high) / 2
    
    def calculate_iv(self, market_price, S, K, T, r, option_type='call', method='auto'):
        """
        Calculate implied volatility using specified method.
        
        Parameters:
        -----------
        method : str
            'newton' - Newton-Raphson method
            'brent' - Brent's method
            'bisection' - Bisection method
            'auto' - Try methods in order of preference
            
        Returns:
        --------
        float
            Implied volatility or np.nan if calculation fails
        """
        if method == 'newton':
            return self.newton_raphson_iv(market_price, S, K, T, r, option_type)
        elif method == 'brent':
            return self.brent_method_iv(market_price, S, K, T, r, option_type)
        elif method == 'bisection':
            return self.bisection_iv(market_price, S, K, T, r, option_type)
        elif method == 'auto':
            # Try methods in order of preference
            iv = self.newton_raphson_iv(market_price, S, K, T, r, option_type)
            if not np.isnan(iv):
                return iv
            
            iv = self.brent_method_iv(market_price, S, K, T, r, option_type)
            if not np.isnan(iv):
                return iv
            
            return self.bisection_iv(market_price, S, K, T, r, option_type)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _check_arbitrage_bounds(self, market_price, S, K, T, r, option_type):
        """
        Check if market price satisfies basic arbitrage bounds.
        
        Returns:
        --------
        bool
            True if price is within arbitrage bounds
        """
        # Calculate intrinsic value
        if option_type.lower() == 'call':
            intrinsic = max(S - K * np.exp(-r * T), 0)
        else:  # put
            intrinsic = max(K * np.exp(-r * T) - S, 0)
        
        # Market price must be at least intrinsic value
        if market_price < intrinsic:
            return False
        
        # Market price cannot exceed certain upper bounds
        if option_type.lower() == 'call':
            upper_bound = S  # Call cannot be worth more than underlying
        else:  # put
            upper_bound = K * np.exp(-r * T)  # Put cannot be worth more than discounted strike
        
        if market_price > upper_bound:
            return False
        
        return True
    
    def calculate_iv_smile(self, market_data, S, T, r):
        """
        Calculate implied volatility smile from market data.
        
        Parameters:
        -----------
        market_data : list of dict
            List containing [{'strike': K, 'price': price, 'type': 'call'/'put'}, ...]
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with strikes and corresponding implied volatilities
        """
        import pandas as pd
        
        results = []
        for data in market_data:
            K = data['strike']
            price = data['price']
            option_type = data['type']
            
            iv = self.calculate_iv(price, S, K, T, r, option_type, method='auto')
            
            results.append({
                'strike': K,
                'option_type': option_type,
                'market_price': price,
                'implied_volatility': iv,
                'moneyness': S / K
            })
        
        return pd.DataFrame(results)


def calculate_iv_surface(market_data, S, r):
    """
    Calculate implied volatility surface from market data across strikes and expirations.
    
    Parameters:
    -----------
    market_data : pandas.DataFrame
        DataFrame with columns: ['strike', 'expiration', 'price', 'option_type']
    S : float
        Current underlying price
    r : float
        Risk-free rate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with IV surface data
    """
    calculator = ImpliedVolatilityCalculator()
    results = []
    
    for _, row in market_data.iterrows():
        K = row['strike']
        T = row['expiration']  # Assuming this is already in years
        price = row['price']
        option_type = row['option_type']
        
        iv = calculator.calculate_iv(price, S, K, T, r, option_type, method='auto')
        
        results.append({
            'strike': K,
            'expiration': T,
            'option_type': option_type,
            'market_price': price,
            'implied_volatility': iv,
            'moneyness': S / K,
            'time_to_expiry': T
        })
    
    import pandas as pd
    return pd.DataFrame(results)