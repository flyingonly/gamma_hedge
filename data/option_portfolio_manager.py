import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

@dataclass
class OptionInfo:
    """Information about a single option"""
    weekly_code: str
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    full_name: str  # e.g., '3CN5/CALL_111.0'
    data_path: str
    data_available: bool = False
    data_size: int = 0
    last_modified: Optional[datetime] = None

@dataclass
class PortfolioConfig:
    """Configuration for option portfolio selection"""
    name: str
    description: str
    selection_criteria: Dict
    max_options: int = 1
    position_weights: Optional[Dict[str, float]] = None
    training_params: Optional[Dict] = None
    dense_mode_params: Optional[Dict] = None
    time_series_params: Optional[Dict] = None

class OptionPortfolioManager:
    """
    Manages option portfolio selection and configuration for training.
    Supports single option initially with architecture for multi-option expansion.
    """
    
    def __init__(self, data_dir: str = "data/preprocessed_greeks"):
        self.data_dir = Path(data_dir)
        self.available_options: List[OptionInfo] = []
        self.portfolio_configs: Dict[str, PortfolioConfig] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize available options inventory
        self._scan_available_options()
        self._load_default_portfolio_configs()
    
    def _scan_available_options(self):
        """Scan the data directory to build inventory of available options"""
        self.available_options = []
        
        if not self.data_dir.exists():
            self.logger.warning(f"Data directory not found: {self.data_dir}")
            return
        
        for weekly_dir in self.data_dir.iterdir():
            if not weekly_dir.is_dir():
                continue
                
            weekly_code = weekly_dir.name
            
            for option_file in weekly_dir.glob("*.npz"):
                try:
                    # Parse option file name: CALL_111.0_greeks.npz -> CALL, 111.0
                    file_stem = option_file.stem  # removes .npz
                    if not file_stem.endswith('_greeks'):
                        continue
                        
                    option_part = file_stem.replace('_greeks', '')  # CALL_111.0
                    parts = option_part.split('_')
                    
                    if len(parts) != 2:
                        continue
                    
                    option_type, strike_str = parts
                    if option_type not in ['CALL', 'PUT']:
                        continue
                    
                    strike = float(strike_str)
                    full_name = f"{weekly_code}/{option_part}"
                    
                    # Get file info
                    file_stats = option_file.stat()
                    
                    option_info = OptionInfo(
                        weekly_code=weekly_code,
                        option_type=option_type,
                        strike=strike,
                        full_name=full_name,
                        data_path=str(option_file),
                        data_available=True,
                        data_size=file_stats.st_size,
                        last_modified=datetime.fromtimestamp(file_stats.st_mtime)
                    )
                    
                    self.available_options.append(option_info)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing option file {option_file}: {e}")
        
        self.logger.info(f"Discovered {len(self.available_options)} available options across "
                        f"{len(set(opt.weekly_code for opt in self.available_options))} weekly codes")
    
    def _load_default_portfolio_configs(self):
        """Load default portfolio configurations"""
        
        # Single ATM Call strategy
        self.portfolio_configs['single_atm_call'] = PortfolioConfig(
            name="Single ATM Call",
            description="Single at-the-money call option for basic delta hedging",
            selection_criteria={
                'option_type': 'CALL',
                'strike_range': {'min_percentile': 45, 'max_percentile': 55},  # Near ATM
                'weekly_codes': ['3CN5', '3IN5', '3MN5'],  # Preferred weeklies
                'min_data_size': 1000  # Minimum file size
            },
            max_options=1,
            position_weights={'selected': 1.0},
            training_params={
                'sequence_length': 100,
                'batch_size': 32,
                'learning_rate': 1e-3
            }
        )
        
        # Single ATM Put strategy  
        self.portfolio_configs['single_atm_put'] = PortfolioConfig(
            name="Single ATM Put",
            description="Single at-the-money put option for basic delta hedging",
            selection_criteria={
                'option_type': 'PUT',
                'strike_range': {'min_percentile': 45, 'max_percentile': 55},
                'weekly_codes': ['3CN5', '3IN5', '3MN5'],
                'min_data_size': 1000
            },
            max_options=1,
            position_weights={'selected': 1.0}
        )
        
        # Best liquidity single option
        self.portfolio_configs['best_liquidity_single'] = PortfolioConfig(
            name="Best Liquidity Single",
            description="Single option with best data quality/size",
            selection_criteria={
                'weekly_codes': ['3CN5', '3CQ5', '3IN5', '3IQ5', '3MN5', '3MQ5'],
                'min_data_size': 5000,
                'sort_by': 'data_size'  # Select largest data file
            },
            max_options=1,
            position_weights={'selected': 1.0}
        )
        
        # Future: Call spread strategy (architecture ready)
        self.portfolio_configs['call_spread'] = PortfolioConfig(
            name="Call Spread",
            description="Long/short call spread strategy (future implementation)",
            selection_criteria={
                'option_type': 'CALL',
                'strike_spread': 2.0,  # 2 point spread
                'weekly_codes': ['3CN5', '3IN5'],
                'min_data_size': 1000
            },
            max_options=2,
            position_weights={'long': 1.0, 'short': -1.0},
            training_params={
                'sequence_length': 100,
                'batch_size': 16  # Smaller batch for complex portfolio
            }
        )
    
    def get_available_weekly_codes(self) -> List[str]:
        """Get list of available weekly codes"""
        return sorted(set(opt.weekly_code for opt in self.available_options))
    
    def get_options_for_weekly(self, weekly_code: str) -> List[OptionInfo]:
        """Get all options for a specific weekly code"""
        return [opt for opt in self.available_options if opt.weekly_code == weekly_code]
    
    def filter_options(self, criteria: Dict) -> List[OptionInfo]:
        """Filter options based on criteria"""
        filtered = self.available_options.copy()
        
        # Filter by option type
        if 'option_type' in criteria:
            option_type = criteria['option_type']
            filtered = [opt for opt in filtered if opt.option_type == option_type]
        
        # Filter by weekly codes
        if 'weekly_codes' in criteria:
            weekly_codes = criteria['weekly_codes']
            filtered = [opt for opt in filtered if opt.weekly_code in weekly_codes]
        
        # Filter by minimum data size
        if 'min_data_size' in criteria:
            min_size = criteria['min_data_size']
            filtered = [opt for opt in filtered if opt.data_size >= min_size]
        
        # Filter by strike range (percentile-based)
        if 'strike_range' in criteria:
            strike_range = criteria['strike_range']
            if filtered:
                strikes = [opt.strike for opt in filtered]
                min_strike = np.percentile(strikes, strike_range.get('min_percentile', 0))
                max_strike = np.percentile(strikes, strike_range.get('max_percentile', 100))
                filtered = [opt for opt in filtered if min_strike <= opt.strike <= max_strike]
        
        # Sort by criteria
        if 'sort_by' in criteria:
            sort_field = criteria['sort_by']
            if sort_field == 'data_size':
                filtered = sorted(filtered, key=lambda x: x.data_size, reverse=True)
            elif sort_field == 'strike':
                filtered = sorted(filtered, key=lambda x: x.strike)
            elif sort_field == 'weekly_code':
                filtered = sorted(filtered, key=lambda x: x.weekly_code)
        
        return filtered
    
    def select_portfolio(self, config_name: str) -> Dict[str, OptionInfo]:
        """
        Select portfolio based on configuration
        
        Returns:
            Dictionary mapping position names to OptionInfo objects
        """
        if config_name not in self.portfolio_configs:
            raise ValueError(f"Unknown portfolio config: {config_name}")
        
        config = self.portfolio_configs[config_name]
        candidates = self.filter_options(config.selection_criteria)
        
        if not candidates:
            # Enhanced error reporting with diagnostic information
            self.logger.error("[ERROR] No options found matching criteria")
            self.logger.error(f"Portfolio config: {config_name}")
            self.logger.error(f"Selection criteria: {config.selection_criteria}")
            self.logger.error(f"Total available options: {len(self.available_options)}")
            
            if self.available_options:
                self.logger.error("Available options summary:")
                weekly_summary = {}
                for opt in self.available_options:
                    if opt.weekly_code not in weekly_summary:
                        weekly_summary[opt.weekly_code] = {'CALL': 0, 'PUT': 0}
                    weekly_summary[opt.weekly_code][opt.option_type] += 1
                
                for weekly_code, counts in weekly_summary.items():
                    self.logger.error(f"  {weekly_code}: {counts['CALL']} calls, {counts['PUT']} puts")
            else:
                self.logger.error("No options data available - check preprocessed_greeks directory")
            
            raise ValueError(f"[ERROR] No options found matching criteria for {config_name}. Check logs for diagnostic information.")
        
        selected_portfolio = {}
        
        if config.max_options == 1:
            # Single option selection with data validation
            best_option = candidates[0]  # Already sorted by filter_options
            
            # Validate minimum data requirements
            min_file_size = 1024  # 1KB minimum
            if best_option.data_size < min_file_size:
                self.logger.error(f"[ERROR] Selected option has insufficient data")
                self.logger.error(f"Option: {best_option.full_name}")
                self.logger.error(f"Data size: {best_option.data_size} bytes (minimum required: {min_file_size})")
                raise ValueError(f"[ERROR] Insufficient data for option {best_option.full_name}")
            
            # Validate NPZ file content
            try:
                import numpy as np
                npz_data = np.load(best_option.data_path)
                required_fields = ['underlying_prices', 'option_prices', 'delta', 'gamma', 'theta', 'vega', 'timestamps']
                missing_fields = [field for field in required_fields if field not in npz_data.files]
                
                if missing_fields:
                    self.logger.error(f"[ERROR] NPZ file missing required fields: {missing_fields}")
                    raise ValueError(f"[ERROR] Invalid NPZ file for {best_option.full_name}")
                
                # Check data length
                prices_length = len(npz_data['underlying_prices'])
                min_data_points = 10  # Reduced minimum for testing - we only have 46 data points
                if prices_length < min_data_points:
                    self.logger.error(f"[ERROR] Insufficient data points for training")
                    self.logger.error(f"Option: {best_option.full_name}")
                    self.logger.error(f"Data points: {prices_length} (minimum required: {min_data_points})")
                    raise ValueError(f"[ERROR] Insufficient data points for {best_option.full_name}")
                
                npz_data.close()
                
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to validate NPZ file: {best_option.data_path}")
                self.logger.error(f"Error details: {str(e)}")
                raise ValueError(f"[ERROR] Invalid data file for {best_option.full_name}")
            
            selected_portfolio['primary'] = best_option
            
            self.logger.info(f"[INFO] Selected single option: {best_option.full_name}")
            self.logger.info(f"  - Strike: {best_option.strike}")
            self.logger.info(f"  - Type: {best_option.option_type}")
            self.logger.info(f"  - Data size: {best_option.data_size:,} bytes")
            self.logger.info(f"  - Data points: {prices_length:,}")
            self.logger.info(f"  - Validation: [PASS]")
        
        else:
            # Multi-option selection (future implementation)
            if config_name == 'call_spread':
                # Example: select two calls with spread
                calls = [opt for opt in candidates if opt.option_type == 'CALL']
                if len(calls) >= 2:
                    # Sort by strike and select spread
                    calls_sorted = sorted(calls, key=lambda x: x.strike)
                    spread_target = config.selection_criteria.get('strike_spread', 2.0)
                    
                    for i in range(len(calls_sorted) - 1):
                        long_call = calls_sorted[i]
                        short_call = calls_sorted[i + 1]
                        
                        if abs(short_call.strike - long_call.strike) <= spread_target + 0.5:
                            selected_portfolio['long'] = long_call
                            selected_portfolio['short'] = short_call
                            break
                    
                    if len(selected_portfolio) == 2:
                        self.logger.info(f"Selected call spread:")
                        self.logger.info(f"  - Long: {selected_portfolio['long'].full_name}")
                        self.logger.info(f"  - Short: {selected_portfolio['short'].full_name}")
            
            if not selected_portfolio:
                self.logger.warning(f"Could not create multi-option portfolio for {config_name}, falling back to single option")
                selected_portfolio['primary'] = candidates[0]
        
        return selected_portfolio
    
    def get_portfolio_positions_dict(self, portfolio: Dict[str, OptionInfo], 
                                   config_name: str) -> Dict[str, float]:
        """
        Convert selected portfolio to positions dictionary for DeltaHedgeDataset
        
        Returns:
            Dictionary mapping option full names to position weights
        """
        config = self.portfolio_configs[config_name]
        positions = {}
        
        for position_name, option_info in portfolio.items():
            # Get weight from config or default to 1.0
            if config.position_weights and position_name in config.position_weights:
                weight = config.position_weights[position_name]
            else:
                weight = 1.0
            
            positions[option_info.full_name] = weight
        
        return positions
    
    def save_portfolio_config(self, config_name: str, config: PortfolioConfig, 
                            output_dir: str = "configs/portfolio_templates"):
        """Save portfolio configuration to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config_file = output_path / f"{config_name}.json"
        
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)
        
        self.logger.info(f"Saved portfolio config to {config_file}")
    
    def load_portfolio_config(self, config_file: str) -> dict:
        """Load portfolio configuration from file or config name"""
        config_path = Path(config_file)
        
        # If it's just a name (no path), look in the templates directory
        if not config_path.exists() and '/' not in config_file and '\\' not in config_file:
            config_path = Path("configs/portfolio_templates") / f"{config_file}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = PortfolioConfig(**config_data)
        config_name = config_path.stem
        
        self.portfolio_configs[config_name] = config
        self.logger.info(f"Loaded portfolio config: {config_name}")
        
        return config_data
    
    def list_available_configs(self) -> List[str]:
        """List all available portfolio configurations"""
        return list(self.portfolio_configs.keys())
    
    def get_config_info(self, config_name: str) -> Dict:
        """Get detailed information about a portfolio configuration"""
        if config_name not in self.portfolio_configs:
            raise ValueError(f"Unknown portfolio config: {config_name}")
        
        config = self.portfolio_configs[config_name]
        candidates = self.filter_options(config.selection_criteria)
        
        return {
            'config': asdict(config),
            'matching_options': len(candidates),
            'sample_options': [opt.full_name for opt in candidates[:5]]
        }
    
    def generate_portfolio_report(self, config_name: str) -> str:
        """Generate a detailed text report for a portfolio configuration"""
        info = self.get_config_info(config_name)
        config = info['config']
        
        report = f"""
Portfolio Configuration Report: {config['name']}
{'='*60}

Description: {config['description']}
Maximum Options: {config['max_options']}

Selection Criteria:
{json.dumps(config['selection_criteria'], indent=2)}

Matching Options Available: {info['matching_options']}

Sample Matching Options:
{chr(10).join(f"  - {opt}" for opt in info['sample_options'])}

Position Weights:
{json.dumps(config.get('position_weights', {}), indent=2)}

Training Parameters:
{json.dumps(config.get('training_params', {}), indent=2)}
"""
        return report

# Global instance for easy access
portfolio_manager = OptionPortfolioManager()