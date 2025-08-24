"""
Black-Scholes Option Pricing Model Module
========================================

This module provides Black-Scholes option pricing and Greeks calculation functionality.
Includes both pricing formulas and Greeks computation for European options.
"""

import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    """
    Black-Scholes option pricing model implementation.
    
    Provides methods for:
    - Option pricing (calls and puts)
    - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
    - Risk management metrics
    """
    
    def __init__(self):
        """Initialize Black-Scholes model."""
        pass
    
    def option_price(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate Black-Scholes option price.
        
        Parameters:
        -----------
        S : float
            Current underlying asset price
        K : float
            Strike price
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate
        sigma : float
            Volatility (annualized)
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            Option theoretical price
        """
        if T <= 0:
            # Handle expiration case - return intrinsic value
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:  # put
                return max(K - S, 0)
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Calculate option price
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def calculate_d1_d2(self, S, K, T, r, sigma):
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.
        
        Returns:
        --------
        tuple
            (d1, d2) values
        """
        if T <= 0:
            return np.nan, np.nan
            
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return d1, d2
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate all option Greeks using Black-Scholes model.
        
        Returns:
        --------
        dict
            Dictionary containing Delta, Gamma, Theta, Vega, Rho
        """
        if T <= 0:
            # Handle expiration case
            if option_type.lower() == 'call':
                delta = 1.0 if S > K else 0.0
            else:  # put
                delta = -1.0 if S < K else 0.0
            
            return {
                'delta': delta,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        # Calculate d1 and d2
        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma)
        
        # Delta calculation
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Gamma calculation (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta calculation
        theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type.lower() == 'call':
            theta = theta_common - r * K * np.exp(-r*T) * norm.cdf(d2)
        else:  # put
            theta = theta_common + r * K * np.exp(-r*T) * norm.cdf(-d2)
        theta = theta / 365  # Convert to daily theta
        
        # Vega calculation (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% volatility change
        
        # Rho calculation
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        else:  # put
            rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def calculate_all(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option price and all Greeks using Black-Scholes model.
        
        Parameters:
        -----------
        S : float
            Current underlying asset price
        K : float
            Strike price
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate
        sigma : float
            Volatility (annualized)
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        dict
            Dictionary containing option_price, Delta, Gamma, Theta, Vega, Rho
        """
        # Calculate option price
        option_price = self.option_price(S, K, T, r, sigma, option_type)
        
        # Calculate Greeks
        greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
        
        # Combine results
        result = {'option_price': option_price}
        result.update(greeks)
        
        return result
    
    def delta(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Delta."""
        return self.calculate_greeks(S, K, T, r, sigma, option_type)['delta']
    
    def gamma(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Gamma."""
        return self.calculate_greeks(S, K, T, r, sigma, option_type)['gamma']
    
    def theta(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Theta."""
        return self.calculate_greeks(S, K, T, r, sigma, option_type)['theta']
    
    def vega(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Vega."""
        return self.calculate_greeks(S, K, T, r, sigma, option_type)['vega']
    
    def rho(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Rho."""
        return self.calculate_greeks(S, K, T, r, sigma, option_type)['rho']
    
    def implied_volatility_vega(self, S, K, T, r, sigma):
        """
        Calculate vega for implied volatility calculations.
        Used in Newton-Raphson method for IV computation.
        """
        if T <= 0:
            return 0
        
        d1, _ = self.calculate_d1_d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)


# Utility functions for the Black-Scholes model
def validate_inputs(S, K, T, r, sigma):
    """
    Validate input parameters for Black-Scholes calculations.
    
    Raises:
    -------
    ValueError
        If any input parameter is invalid
    """
    if S <= 0:
        raise ValueError("Underlying price must be positive")
    if K <= 0:
        raise ValueError("Strike price must be positive")
    if T < 0:
        raise ValueError("Time to expiration cannot be negative")
    if sigma < 0:
        raise ValueError("Volatility cannot be negative")
    # Note: r can be negative (negative interest rates)


def calculate_moneyness(S, K):
    """
    Calculate option moneyness.
    
    Returns:
    --------
    str
        'ITM' (in-the-money), 'ATM' (at-the-money), or 'OTM' (out-of-the-money)
    """
    ratio = S / K
    if abs(ratio - 1.0) < 0.01:  # Within 1% of ATM
        return 'ATM'
    elif ratio > 1.0:
        return 'ITM'  # For calls
    else:
        return 'OTM'  # For calls


def calculate_time_value(option_price, S, K, option_type='call'):
    """
    Calculate option time value (extrinsic value).
    
    Returns:
    --------
    float
        Time value of the option
    """
    if option_type.lower() == 'call':
        intrinsic_value = max(S - K, 0)
    else:  # put
        intrinsic_value = max(K - S, 0)
    
    return option_price - intrinsic_value