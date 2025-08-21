"""
Vectorized Black-Scholes Calculations
====================================

High-performance vectorized implementation of Black-Scholes formulas
optimized for dense data processing with 20K+ time points.
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def black_scholes_delta(S: Union[float, np.ndarray],
                       K: Union[float, np.ndarray], 
                       T: Union[float, np.ndarray],
                       r: Union[float, np.ndarray],
                       sigma: Union[float, np.ndarray],
                       option_type: str = 'call') -> Union[float, np.ndarray]:
    """
    Vectorized Black-Scholes Delta calculation
    
    Args:
        S: Underlying price (can be array)
        K: Strike price (can be array)
        T: Time to expiration in years (can be array)
        r: Risk-free rate (can be array)
        sigma: Implied volatility (can be array)
        option_type: 'call' or 'put'
    
    Returns:
        Delta values (same shape as input arrays)
    """
    # Convert inputs to numpy arrays for vectorization
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    # Avoid division by zero when T or sigma is very small
    T = np.maximum(T, 1e-10)  # Minimum time to expiration
    sigma = np.maximum(sigma, 1e-6)  # Minimum volatility
    
    # Calculate d1
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Calculate delta based on option type
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    elif option_type.lower() == 'put':
        delta = norm.cdf(d1) - 1.0
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")
    
    return delta


def black_scholes_gamma(S: Union[float, np.ndarray],
                       K: Union[float, np.ndarray],
                       T: Union[float, np.ndarray], 
                       r: Union[float, np.ndarray],
                       sigma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Vectorized Black-Scholes Gamma calculation
    
    Args:
        S: Underlying price (can be array)
        K: Strike price (can be array)
        T: Time to expiration in years (can be array)
        r: Risk-free rate (can be array)
        sigma: Implied volatility (can be array)
    
    Returns:
        Gamma values (same shape as input arrays)
    """
    # Convert inputs to numpy arrays
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-6)
    
    # Calculate d1
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Calculate gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return gamma


def black_scholes_theta(S: Union[float, np.ndarray],
                       K: Union[float, np.ndarray],
                       T: Union[float, np.ndarray],
                       r: Union[float, np.ndarray], 
                       sigma: Union[float, np.ndarray],
                       option_type: str = 'call') -> Union[float, np.ndarray]:
    """
    Vectorized Black-Scholes Theta calculation
    
    Args:
        S: Underlying price (can be array)
        K: Strike price (can be array)
        T: Time to expiration in years (can be array)
        r: Risk-free rate (can be array)
        sigma: Implied volatility (can be array)
        option_type: 'call' or 'put'
    
    Returns:
        Theta values (same shape as input arrays)
    """
    # Convert inputs to numpy arrays
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-6)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Common theta components
    theta_common = (-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    if option_type.lower() == 'call':
        theta = theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        theta = theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")
    
    # Convert to per-day theta (divide by 365)
    theta = theta / 365.0
    
    return theta


def black_scholes_vega(S: Union[float, np.ndarray],
                      K: Union[float, np.ndarray],
                      T: Union[float, np.ndarray],
                      r: Union[float, np.ndarray],
                      sigma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Vectorized Black-Scholes Vega calculation
    
    Args:
        S: Underlying price (can be array)
        K: Strike price (can be array)
        T: Time to expiration in years (can be array)
        r: Risk-free rate (can be array)
        sigma: Implied volatility (can be array)
    
    Returns:
        Vega values (same shape as input arrays)
    """
    # Convert inputs to numpy arrays
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-6)
    
    # Calculate d1
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Calculate vega (same for calls and puts)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    # Convert to per-1% volatility change (divide by 100)
    vega = vega / 100.0
    
    return vega


def calculate_all_greeks(S: Union[float, np.ndarray],
                        K: Union[float, np.ndarray],
                        T: Union[float, np.ndarray],
                        r: Union[float, np.ndarray],
                        sigma: Union[float, np.ndarray],
                        option_type: str = 'call') -> dict:
    """
    Calculate all Greeks in one vectorized operation for efficiency
    
    Args:
        S: Underlying price (can be array)
        K: Strike price (can be array)
        T: Time to expiration in years (can be array)
        r: Risk-free rate (can be array)
        sigma: Implied volatility (can be array)
        option_type: 'call' or 'put'
    
    Returns:
        Dictionary containing all Greeks
    """
    # Convert inputs to numpy arrays
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-6)
    
    # Calculate d1 and d2 once for efficiency
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate all Greeks
    greeks = {}
    
    # Delta
    if option_type.lower() == 'call':
        greeks['delta'] = norm.cdf(d1)
    elif option_type.lower() == 'put':
        greeks['delta'] = norm.cdf(d1) - 1.0
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")
    
    # Gamma (same for calls and puts)
    greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    theta_common = (-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type.lower() == 'call':
        greeks['theta'] = (theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:  # put
        greeks['theta'] = (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    
    # Vega (same for calls and puts)
    greeks['vega'] = (S * norm.pdf(d1) * np.sqrt(T)) / 100.0
    
    return greeks


def interpolate_implied_volatility(option_timestamps: np.ndarray,
                                  option_iv: np.ndarray,
                                  target_timestamps: np.ndarray,
                                  method: str = 'linear',
                                  fill_value: str = 'extrapolate') -> np.ndarray:
    """
    Interpolate implied volatility to target timestamps
    
    Args:
        option_timestamps: Original timestamp array (sparse)
        option_iv: Implied volatility values at option timestamps
        target_timestamps: Target timestamp array (dense)
        method: Interpolation method ('linear', 'cubic', 'nearest')
        fill_value: How to handle out-of-bounds values
    
    Returns:
        Interpolated implied volatility at target timestamps
    """
    from scipy.interpolate import interp1d
    
    # Input validation
    if len(option_timestamps) != len(option_iv):
        raise ValueError("option_timestamps and option_iv must have same length")
    
    if len(option_timestamps) < 2:
        raise ValueError("Need at least 2 data points for interpolation")
    
    # Sort by timestamps to ensure monotonic order
    sort_indices = np.argsort(option_timestamps)
    option_timestamps = option_timestamps[sort_indices]
    option_iv = option_iv[sort_indices]
    
    # Remove duplicates while preserving order
    unique_indices = np.concatenate(([True], np.diff(option_timestamps) != 0))
    option_timestamps = option_timestamps[unique_indices]
    option_iv = option_iv[unique_indices]
    
    # Create interpolation function
    try:
        interpolator = interp1d(
            option_timestamps, 
            option_iv,
            kind=method,
            fill_value=fill_value,
            assume_sorted=True
        )
        
        # Perform interpolation
        interpolated_iv = interpolator(target_timestamps)
        
        # Ensure positive volatility values
        interpolated_iv = np.maximum(interpolated_iv, 1e-6)
        
        logger.info(f"Successfully interpolated IV from {len(option_timestamps)} to {len(target_timestamps)} points")
        
        return interpolated_iv
        
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        # Fallback: use nearest neighbor or constant fill
        if len(option_iv) > 0:
            return np.full(len(target_timestamps), np.mean(option_iv))
        else:
            return np.full(len(target_timestamps), 0.2)  # Default 20% volatility


def validate_bs_inputs(S, K, T, r, sigma, option_type: str = 'call') -> bool:
    """
    Validate Black-Scholes input parameters
    
    Args:
        S: Underlying price
        K: Strike price  
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        option_type: Option type
    
    Returns:
        True if all inputs are valid
    """
    try:
        # Convert to arrays for validation
        S = np.asarray(S)
        K = np.asarray(K)
        T = np.asarray(T)
        r = np.asarray(r)
        sigma = np.asarray(sigma)
        
        # Check for non-finite values
        if not (np.all(np.isfinite(S)) and np.all(np.isfinite(K)) and 
                np.all(np.isfinite(T)) and np.all(np.isfinite(r)) and 
                np.all(np.isfinite(sigma))):
            logger.warning("Non-finite values detected in BS inputs")
            return False
        
        # Check for negative values where inappropriate
        if np.any(S <= 0):
            logger.warning("Non-positive underlying prices detected")
            return False
            
        if np.any(K <= 0):
            logger.warning("Non-positive strike prices detected")
            return False
            
        if np.any(T < 0):
            logger.warning("Negative time to expiration detected")
            return False
            
        if np.any(sigma <= 0):
            logger.warning("Non-positive volatility detected")
            return False
        
        # Check option type
        if option_type.lower() not in ['call', 'put']:
            logger.warning(f"Invalid option type: {option_type}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        return False


# Performance testing utilities
def benchmark_vectorized_calculation(n_points: int = 20000) -> dict:
    """
    Benchmark vectorized Black-Scholes calculations
    
    Args:
        n_points: Number of data points to test
    
    Returns:
        Performance statistics
    """
    import time
    
    # Generate test data
    S = np.random.uniform(100, 120, n_points)
    K = np.full(n_points, 110.0)
    T = np.random.uniform(0.01, 1.0, n_points)
    r = np.full(n_points, 0.05)
    sigma = np.random.uniform(0.1, 0.5, n_points)
    
    # Benchmark individual calculations
    results = {}
    
    # Delta calculation
    start_time = time.time()
    delta = black_scholes_delta(S, K, T, r, sigma, 'call')
    results['delta_time'] = time.time() - start_time
    
    # All Greeks calculation
    start_time = time.time()
    all_greeks = calculate_all_greeks(S, K, T, r, sigma, 'call')
    results['all_greeks_time'] = time.time() - start_time
    
    # Calculate performance metrics
    results['points_per_second_delta'] = n_points / results['delta_time']
    results['points_per_second_all'] = n_points / results['all_greeks_time']
    results['n_points'] = n_points
    
    logger.info(f"Benchmark results for {n_points} points:")
    logger.info(f"  Delta calculation: {results['delta_time']:.4f}s ({results['points_per_second_delta']:.0f} pts/s)")
    logger.info(f"  All Greeks: {results['all_greeks_time']:.4f}s ({results['points_per_second_all']:.0f} pts/s)")
    
    return results


if __name__ == "__main__":
    # Run basic performance test
    print("Running vectorized Black-Scholes benchmark...")
    benchmark_results = benchmark_vectorized_calculation(20000)
    print(f"Performance test completed: {benchmark_results['points_per_second_all']:.0f} calculations/second")