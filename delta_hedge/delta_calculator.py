"""
Delta Calculator Module
======================

Core delta calculation engine for options portfolios.
Implements delta hedging strategies and weight calculation for policy learning integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

from .config import DeltaHedgeConfig
from .data_loader import OptionsDataLoader
from .bs_integration import EnhancedBlackScholesCalculator

# Setup logging
logger = logging.getLogger(__name__)


class DeltaCalculator:
    """
    Core delta calculation engine for options portfolios.
    
    Provides:
    - Real-time delta calculation for individual options and portfolios
    - Delta hedging weight calculation
    - Dynamic rebalancing strategies
    - Risk metrics and portfolio optimization
    """
    
    def __init__(self, config: DeltaHedgeConfig):
        """
        Initialize delta calculator.
        
        Args:
            config: Delta hedging configuration
        """
        self.config = config
        self.data_loader = OptionsDataLoader(config)
        self.bs_calculator = EnhancedBlackScholesCalculator(config)
        
        # State variables
        self.current_portfolio = {}
        self.last_rebalance_time = None
        self.cumulative_hedging_cost = 0.0
        
        logger.info("Delta calculator initialized")
    
    def calculate_single_option_delta(self, S: float, K: float, T: float,
                                    r: float, sigma: float, option_type: str) -> float:
        """
        Calculate delta for a single option.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option delta
        """
        greeks = self.bs_calculator.calculate_greeks_batch(
            S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type
        )
        return float(greeks['delta'][0])
    
    def calculate_portfolio_delta(self, portfolio_positions: Dict[str, float],
                                options_data: Dict[str, Dict[str, np.ndarray]],
                                underlying_price: float,
                                timestamp: datetime,
                                contract_info: Optional[Dict] = None) -> Dict[str, Union[float, Dict]]:
        """
        Calculate portfolio delta and detailed metrics.
        
        Args:
            portfolio_positions: Dictionary mapping option IDs to position sizes
            options_data: Current options market data
            underlying_price: Current underlying asset price
            timestamp: Current timestamp
            contract_info: Contract specifications including expiry_date
            
        Returns:
            Dictionary with portfolio delta and detailed breakdown
        """
        portfolio_delta = 0.0
        position_details = {}
        total_notional = 0.0
        
        for option_id, position_size in portfolio_positions.items():
            if option_id not in options_data or position_size == 0:
                continue
            
            option_info = options_data[option_id]
            
            # Get current option parameters
            strike = float(option_info['strikes'][-1])  # Latest strike
            option_type = option_info['option_types'][0]  # Option type
            
            # Calculate time to expiry using actual contract info
            time_to_expiry = self._calculate_time_to_expiry(timestamp, contract_info)
            
            # Skip calculation if option has expired (configurable)
            if (self.config.delta_calculation.enable_expiry_checking and 
                self.config.delta_calculation.skip_expired_options and 
                time_to_expiry < 0):
                logger.info(f"Skipping expired option {option_id} at {timestamp}")
                position_details[option_id] = {
                    'position_size': position_size,
                    'option_delta': 0.0,
                    'position_delta': 0.0,
                    'strike': strike,
                    'option_type': option_type,
                    'implied_vol': 0.0,
                    'option_price': 0.0,
                    'notional': 0.0,
                    'expired': True
                }
                continue
            
            # Get volatility with improved estimation
            sigma = self._estimate_volatility(option_info, underlying_price)
            
            # Calculate option delta
            option_delta = self.calculate_single_option_delta(
                S=underlying_price,
                K=strike,
                T=time_to_expiry,
                r=self.config.black_scholes.risk_free_rate,
                sigma=sigma,
                option_type=option_type
            )
            
            # Calculate position delta
            position_delta = position_size * option_delta
            portfolio_delta += position_delta
            
            # Calculate notional value
            option_price = option_info.get('market_prices', [strike * 0.05])[-1]  # Fallback price
            position_notional = abs(position_size) * option_price
            total_notional += position_notional
            
            position_details[option_id] = {
                'position_size': position_size,
                'option_delta': option_delta,
                'position_delta': position_delta,
                'strike': strike,
                'option_type': option_type,
                'implied_vol': sigma,
                'option_price': option_price,
                'notional': position_notional,
                'expired': False
            }
        
        return {
            'portfolio_delta': portfolio_delta,
            'total_notional': total_notional,
            'delta_per_dollar': portfolio_delta / max(total_notional, 1.0),
            'position_count': len([p for p in portfolio_positions.values() if p != 0]),
            'timestamp': timestamp,
            'underlying_price': underlying_price,
            'position_details': position_details
        }
    
    def calculate_hedging_weights(self, portfolio_delta: float,
                                underlying_price: float,
                                hedge_method: str = "delta_neutral") -> Dict[str, float]:
        """
        Calculate hedging weights for delta neutrality.
        
        Args:
            portfolio_delta: Current portfolio delta
            underlying_price: Current underlying price
            hedge_method: Hedging method ('delta_neutral', 'risk_adjusted', 'cost_optimal')
            
        Returns:
            Dictionary with hedging information
        """
        # Base hedge ratio (negative of portfolio delta for neutrality)
        base_hedge_ratio = -portfolio_delta
        
        # Apply hedging method adjustments
        if hedge_method == "delta_neutral":
            hedge_ratio = base_hedge_ratio
        elif hedge_method == "risk_adjusted":
            # Apply risk adjustment factor
            risk_factor = self._calculate_risk_adjustment_factor(portfolio_delta)
            hedge_ratio = base_hedge_ratio * risk_factor
        elif hedge_method == "cost_optimal":
            # Consider transaction costs
            cost_factor = self._calculate_cost_adjustment_factor(portfolio_delta, underlying_price)
            hedge_ratio = base_hedge_ratio * cost_factor
        else:
            raise ValueError(f"Unknown hedge method: {hedge_method}")
        
        # Apply constraints
        max_hedge_ratio = self.config.delta_calculation.max_hedge_ratio
        hedge_ratio = np.clip(hedge_ratio, -max_hedge_ratio, max_hedge_ratio)
        
        # Calculate hedge notional
        hedge_notional = hedge_ratio * underlying_price
        
        # Check minimum hedge threshold
        min_notional = self.config.delta_calculation.min_hedge_notional
        if abs(hedge_notional) < min_notional:
            hedge_ratio = 0.0
            hedge_notional = 0.0
        
        return {
            'hedge_ratio': hedge_ratio,
            'hedge_notional': hedge_notional,
            'base_hedge_ratio': base_hedge_ratio,
            'hedge_method': hedge_method,
            'underlying_price': underlying_price,
            'constraints_applied': abs(base_hedge_ratio) != abs(hedge_ratio)
        }
    
    def should_rebalance(self, current_delta: float, last_delta: float,
                        current_time: datetime) -> Dict[str, bool]:
        """
        Determine if portfolio should be rebalanced based on delta drift and time.
        
        Args:
            current_delta: Current portfolio delta
            last_delta: Last recorded portfolio delta
            current_time: Current timestamp
            
        Returns:
            Dictionary with rebalancing decision and reasons
        """
        rebalance_reasons = {
            'delta_threshold': False,
            'time_threshold': False,
            'risk_limit': False
        }
        
        # Check delta threshold
        delta_change = abs(current_delta - last_delta)
        delta_threshold = self.config.delta_calculation.rebalance_threshold
        if delta_change >= delta_threshold:
            rebalance_reasons['delta_threshold'] = True
        
        # Check time threshold (e.g., rebalance at least daily)
        if self.last_rebalance_time is not None:
            time_since_rebalance = (current_time - self.last_rebalance_time).total_seconds() / 3600
            if time_since_rebalance >= 24:  # 24 hours
                rebalance_reasons['time_threshold'] = True
        else:
            rebalance_reasons['time_threshold'] = True  # First rebalance
        
        # Check risk limits
        max_delta = self.config.delta_calculation.max_portfolio_delta
        if abs(current_delta) >= max_delta:
            rebalance_reasons['risk_limit'] = True
        
        should_rebalance = any(rebalance_reasons.values())
        
        return {
            'should_rebalance': should_rebalance,
            'reasons': rebalance_reasons,
            'delta_change': delta_change,
            'current_delta': current_delta,
            'last_delta': last_delta
        }
    
    def calculate_dynamic_hedge_weights(self, underlying_prices: np.ndarray,
                                      options_timeseries: Dict[str, Dict[str, np.ndarray]],
                                      portfolio_positions: Dict[str, float],
                                      timestamps: np.ndarray,
                                      contract_info: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Calculate dynamic hedging weights over a time series.
        
        Args:
            underlying_prices: Array of underlying prices over time
            options_timeseries: Time series of options data
            portfolio_positions: Static portfolio positions
            timestamps: Array of timestamps
            contract_info: Contract specifications including expiry_date
            
        Returns:
            Dictionary with time series of hedging weights and metrics
        """
        n_timesteps = len(underlying_prices)
        
        # Initialize output arrays
        hedge_ratios = np.zeros(n_timesteps)
        portfolio_deltas = np.zeros(n_timesteps)
        rebalance_flags = np.zeros(n_timesteps, dtype=bool)
        hedging_costs = np.zeros(n_timesteps)
        
        last_hedge_ratio = 0.0
        last_delta = 0.0
        
        for i, (price, timestamp) in enumerate(zip(underlying_prices, timestamps)):
            # Extract current options data for this timestep
            current_options_data = {}
            for option_id, option_timeseries in options_timeseries.items():
                current_options_data[option_id] = {
                    key: values[i:i+1] if isinstance(values, np.ndarray) else values
                    for key, values in option_timeseries.items()
                }
            
            # Calculate current portfolio delta
            portfolio_metrics = self.calculate_portfolio_delta(
                portfolio_positions=portfolio_positions,
                options_data=current_options_data,
                underlying_price=price,
                timestamp=pd.to_datetime(timestamp),
                contract_info=contract_info
            )
            
            current_delta = portfolio_metrics['portfolio_delta']
            portfolio_deltas[i] = current_delta
            
            # Check if rebalancing is needed
            rebalance_decision = self.should_rebalance(
                current_delta=current_delta,
                last_delta=last_delta,
                current_time=pd.to_datetime(timestamp)
            )
            
            if rebalance_decision['should_rebalance']:
                # Calculate new hedge weights
                hedge_weights = self.calculate_hedging_weights(
                    portfolio_delta=current_delta,
                    underlying_price=price,
                    hedge_method="delta_neutral"  # Use valid hedge method
                )
                
                new_hedge_ratio = hedge_weights['hedge_ratio']
                
                # Calculate hedging cost (simplified)
                hedging_cost = abs(new_hedge_ratio - last_hedge_ratio) * price * 0.001  # 0.1% transaction cost
                hedging_costs[i] = hedging_cost
                self.cumulative_hedging_cost += hedging_cost
                
                hedge_ratios[i] = new_hedge_ratio
                rebalance_flags[i] = True
                last_hedge_ratio = new_hedge_ratio
                
                # Update rebalance time
                if isinstance(timestamp, (int, float)):
                    self.last_rebalance_time = pd.to_datetime(timestamp, unit='s')
                else:
                    self.last_rebalance_time = pd.to_datetime(timestamp)
            else:
                # No rebalancing - keep previous hedge ratio
                hedge_ratios[i] = last_hedge_ratio
                rebalance_flags[i] = False
            
            last_delta = current_delta
        
        # Calculate additional metrics
        hedge_effectiveness = self._calculate_hedge_effectiveness(
            portfolio_deltas, hedge_ratios, underlying_prices
        )
        
        return {
            'hedge_ratios': hedge_ratios,
            'portfolio_deltas': portfolio_deltas,
            'rebalance_flags': rebalance_flags,
            'hedging_costs': hedging_costs,
            'cumulative_cost': np.cumsum(hedging_costs),
            'hedge_effectiveness': hedge_effectiveness,
            'total_rebalances': int(np.sum(rebalance_flags)),
            'final_hedge_ratio': hedge_ratios[-1],
            'final_portfolio_delta': portfolio_deltas[-1]
        }
    
    def optimize_portfolio_weights(self, available_options: List[str],
                                 target_characteristics: Dict[str, float],
                                 constraints: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize portfolio weights to achieve target risk characteristics.
        
        Args:
            available_options: List of available option IDs
            target_characteristics: Target portfolio characteristics (e.g., target_delta, target_gamma)
            constraints: Weight and risk constraints
            
        Returns:
            Dictionary mapping option IDs to optimal weights
        """
        # This is a placeholder for portfolio optimization
        # In practice, this would use optimization techniques like quadratic programming
        
        target_delta = target_characteristics.get('target_delta', 0.0)
        max_weight = constraints.get('max_individual_weight', 0.5)
        
        # Simple equal-weighting approach for demonstration
        if not available_options:
            return {}
        
        equal_weight = min(1.0 / len(available_options), max_weight)
        weights = {option_id: equal_weight for option_id in available_options}
        
        logger.info(f"Optimized weights for {len(available_options)} options")
        return weights
    
    def _calculate_risk_adjustment_factor(self, portfolio_delta: float) -> float:
        """Calculate risk adjustment factor for hedging"""
        # Higher risk adjustment for larger deltas
        risk_factor = 1.0 - 0.05 * min(abs(portfolio_delta) / 100, 1.0)
        return max(0.8, risk_factor)  # Minimum 80% hedge
    
    def _calculate_cost_adjustment_factor(self, portfolio_delta: float, 
                                        underlying_price: float) -> float:
        """Calculate cost adjustment factor for hedging"""
        # Reduce hedging for small deltas to minimize transaction costs
        notional_delta = abs(portfolio_delta * underlying_price)
        min_notional = self.config.delta_calculation.min_hedge_notional
        
        if notional_delta < min_notional:
            return 0.0  # No hedging below minimum
        elif notional_delta < min_notional * 2:
            return 0.5  # Partial hedging
        else:
            return 1.0  # Full hedging
    
    def _calculate_hedge_effectiveness(self, portfolio_deltas: np.ndarray,
                                     hedge_ratios: np.ndarray,
                                     underlying_prices: np.ndarray) -> Dict[str, float]:
        """Calculate hedge effectiveness metrics"""
        if len(portfolio_deltas) < 2:
            return {'effectiveness': 0.0, 'correlation': 0.0, 'hedge_ratio_std': 0.0}
        
        # Calculate hedged portfolio delta
        hedged_deltas = portfolio_deltas + hedge_ratios
        
        # Calculate effectiveness as reduction in delta volatility
        unhedged_std = np.std(portfolio_deltas)
        hedged_std = np.std(hedged_deltas)
        
        effectiveness = max(0.0, 1.0 - (hedged_std / max(unhedged_std, 1e-8)))
        
        # Calculate correlation between hedge and portfolio delta
        if np.std(hedge_ratios) > 1e-8 and np.std(portfolio_deltas) > 1e-8:
            correlation = np.corrcoef(hedge_ratios, portfolio_deltas)[0, 1]
        else:
            correlation = 0.0
        
        return {
            'effectiveness': effectiveness,
            'correlation': correlation,
            'hedge_ratio_std': np.std(hedge_ratios),
            'unhedged_delta_std': unhedged_std,
            'hedged_delta_std': hedged_std
        }
    
    def _calculate_time_to_expiry(self, current_time: datetime, contract_info: Optional[Dict]) -> float:
        """
        Calculate time to expiry in years using actual contract information.
        
        Args:
            current_time: Current timestamp
            contract_info: Contract specifications with expiry_date
            
        Returns:
            Time to expiry in years, or -1.0 if option has expired
        """
        if contract_info and 'expiry_date' in contract_info:
            try:
                expiry_date = pd.to_datetime(contract_info['expiry_date'])
                time_diff = expiry_date - current_time
                days_to_expiry = time_diff.total_seconds() / (24 * 3600)
                
                # Check if option has expired
                if days_to_expiry <= 0:
                    logger.info(f"Option expired on {expiry_date}, current time: {current_time}")
                    return -1.0  # Special value indicating expiry
                
                # Use consistent business days per year from config
                years_to_expiry = days_to_expiry / self.config.black_scholes.business_days_per_year
                return max(self.config.black_scholes.min_time_to_expiry, years_to_expiry)
            except Exception as e:
                logger.warning(f"Failed to parse expiry date from contract info: {e}")
        
        # Fallback to default if no contract info available
        logger.warning("Using default time to expiry - contract info not available")
        return max(self.config.black_scholes.min_time_to_expiry, 30/252)  # 30 business days
    
    def _estimate_volatility(self, option_info: Dict[str, np.ndarray], underlying_price: float) -> float:
        """
        Estimate volatility with improved logic and validation.
        
        Args:
            option_info: Option market data
            underlying_price: Current underlying price
            
        Returns:
            Estimated volatility
        """
        # Priority 1: Use implied volatility if available and valid
        if 'implied_volatilities' in option_info and len(option_info['implied_volatilities']) > 0:
            iv = float(option_info['implied_volatilities'][-1])
            if self._validate_volatility(iv):
                return iv
        
        # Priority 2: Calculate historical volatility from market prices
        if 'market_prices' in option_info and len(option_info['market_prices']) > 5:
            try:
                hist_vol = self._calculate_historical_volatility_from_options(option_info, underlying_price)
                if self._validate_volatility(hist_vol):
                    return hist_vol
            except Exception as e:
                logger.warning(f"Failed to calculate historical volatility: {e}")
        
        # Priority 3: Use asset-specific default volatility based on underlying
        default_vol = self._get_asset_default_volatility(underlying_price)
        
        logger.info(f"Using default volatility {default_vol:.3f} for option")
        return default_vol
    
    def _validate_volatility(self, volatility: float) -> bool:
        """Validate volatility is within reasonable bounds"""
        return (self.config.black_scholes.min_volatility <= volatility <= 
                self.config.black_scholes.max_volatility and 
                not np.isnan(volatility) and np.isfinite(volatility))
    
    def _calculate_historical_volatility_from_options(self, option_info: Dict[str, np.ndarray], 
                                                    underlying_price: float) -> float:
        """Calculate implied volatility from recent option price movements"""
        prices = option_info['market_prices']
        if len(prices) < 5:
            raise ValueError("Insufficient price data")
        
        # Simple historical volatility estimation from option prices
        returns = np.diff(prices) / prices[:-1]
        vol = np.std(returns) * np.sqrt(self.config.black_scholes.business_days_per_year)
        
        # Adjust for option leverage (approximate)
        strike = float(option_info['strikes'][-1])
        moneyness = underlying_price / strike
        leverage_adjustment = min(2.0, max(0.5, abs(moneyness - 1.0) + 1.0))
        
        return vol / leverage_adjustment
    
    def _get_asset_default_volatility(self, underlying_price: float) -> float:
        """Get asset-specific default volatility based on price level and asset type"""
        # Bond futures typically have lower volatility
        if underlying_price < 150:  # Likely bond futures
            return 0.15  # 15% default for bonds
        else:  # Equity index futures or higher-priced assets
            return 0.25  # 25% default for equities
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the hedging strategy"""
        return {
            'cumulative_hedging_cost': self.cumulative_hedging_cost,
            'last_rebalance_time': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None,
            'portfolio_positions_count': len(self.current_portfolio)
        }
    
    def reset_state(self):
        """Reset calculator state for new calculation session"""
        self.current_portfolio = {}
        self.last_rebalance_time = None
        self.cumulative_hedging_cost = 0.0
        logger.info("Delta calculator state reset")


class PortfolioRiskManager:
    """
    Portfolio risk management utilities for delta hedging.
    """
    
    def __init__(self, config: DeltaHedgeConfig):
        self.config = config
    
    def calculate_var(self, portfolio_deltas: np.ndarray, 
                     underlying_returns: np.ndarray,
                     confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk for the portfolio"""
        if len(portfolio_deltas) != len(underlying_returns):
            raise ValueError("Deltas and returns must have same length")
        
        # Calculate portfolio PnL
        portfolio_pnl = portfolio_deltas[:-1] * np.diff(underlying_returns)  # Approximate delta PnL
        
        if len(portfolio_pnl) == 0:
            return 0.0
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(portfolio_pnl, var_percentile)
        
        return abs(var)
    
    def check_risk_limits(self, portfolio_metrics: Dict) -> Dict[str, bool]:
        """Check if portfolio violates risk limits"""
        limits_ok = {}
        
        # Check delta limit
        max_delta = self.config.delta_calculation.max_portfolio_delta
        limits_ok['delta_limit'] = abs(portfolio_metrics['portfolio_delta']) <= max_delta
        
        # Check individual position limits
        max_individual_weight = self.config.delta_calculation.max_individual_weight
        position_details = portfolio_metrics.get('position_details', {})
        
        if position_details:
            total_notional = portfolio_metrics.get('total_notional', 1.0)
            max_position_weight = max(
                pos['notional'] / max(total_notional, 1.0) 
                for pos in position_details.values()
            )
            limits_ok['position_concentration'] = max_position_weight <= max_individual_weight
        else:
            limits_ok['position_concentration'] = True
        
        return limits_ok