"""
Simplified Portfolio Management for Gamma Hedge

This module provides simplified portfolio management functionality,
replacing the more complex option_portfolio_manager.py with a
cleaner, more focused approach aligned with the new architecture.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from core.interfaces import PortfolioPosition, create_portfolio_from_dict, portfolio_to_dict
from core.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioTemplate:
    """Template for portfolio configurations"""
    name: str
    description: str
    positions: Dict[str, float]  # option_key -> position_size
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioTemplate':
        """Create portfolio template from dictionary"""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def get_portfolio_positions(self) -> List[PortfolioPosition]:
        """Get portfolio positions as list of PortfolioPosition objects"""
        return create_portfolio_from_dict(self.positions)


class PortfolioManager:
    """
    Simplified portfolio manager focused on essential functionality
    
    Responsibilities:
    - Load portfolio templates from configuration files
    - Validate portfolio positions against available data
    - Provide portfolio information for training
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.templates: Dict[str, PortfolioTemplate] = {}
        
        # Load portfolio templates
        self._load_portfolio_templates()
        
        logger.info(f"Loaded {len(self.templates)} portfolio templates")
    
    def _load_portfolio_templates(self):
        """Load portfolio templates from configuration files"""
        template_dir = Path("configs/portfolio_templates")
        
        if not template_dir.exists():
            logger.warning(f"Portfolio templates directory not found: {template_dir}")
            # Create default template
            self._create_default_template()
            return
        
        # Load all JSON templates
        for template_file in template_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                
                template = PortfolioTemplate.from_dict(template_data)
                self.templates[template.name] = template
                
                logger.info(f"Loaded portfolio template: {template.name}")
                
            except Exception as e:
                logger.error(f"Failed to load portfolio template {template_file}: {e}")
    
    def _create_default_template(self):
        """Create a default portfolio template"""
        default_positions = {}
        
        # Use first weekly code from configuration
        if self.config.delta_hedge.weekly_codes:
            weekly_code = self.config.delta_hedge.weekly_codes[0]
            default_positions[f"{weekly_code}/CALL_111.0"] = 1.0
        
        default_template = PortfolioTemplate(
            name="default",
            description="Default single option portfolio",
            positions=default_positions,
            metadata={
                "created": datetime.now().isoformat(),
                "source": "auto_generated"
            }
        )
        
        self.templates["default"] = default_template
        logger.info("Created default portfolio template")
    
    def get_portfolio_template(self, name: str) -> Optional[PortfolioTemplate]:
        """Get portfolio template by name"""
        return self.templates.get(name)
    
    def list_available_templates(self) -> List[str]:
        """List all available portfolio template names"""
        return list(self.templates.keys())
    
    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a portfolio template"""
        template = self.get_portfolio_template(name)
        if not template:
            return None
        
        info = {
            'name': template.name,
            'description': template.description,
            'num_positions': len(template.positions),
            'positions': template.positions,
            'metadata': template.metadata
        }
        
        # Add data availability information
        info['data_availability'] = self._check_data_availability(template)
        
        return info
    
    def _check_data_availability(self, template: PortfolioTemplate) -> Dict[str, Any]:
        """Check data availability for portfolio positions"""
        availability = {
            'total_positions': len(template.positions),
            'available_positions': 0,
            'missing_data': [],
            'available_data': []
        }
        
        base_path = Path("data/preprocessed_greeks")
        
        for option_key in template.positions.keys():
            try:
                # Parse option key: "3CN5/CALL_111.0"
                weekly_code, option_spec = option_key.split('/')
                
                # Check if Greeks data exists
                file_path = base_path / weekly_code / f"{option_spec}_greeks.npz"
                
                if file_path.exists():
                    availability['available_positions'] += 1
                    availability['available_data'].append(option_key)
                else:
                    availability['missing_data'].append(option_key)
                    
            except Exception as e:
                logger.warning(f"Error checking data for {option_key}: {e}")
                availability['missing_data'].append(option_key)
        
        # Calculate availability percentage
        if availability['total_positions'] > 0:
            availability['availability_percentage'] = (
                availability['available_positions'] / availability['total_positions'] * 100
            )
        else:
            availability['availability_percentage'] = 0.0
        
        return availability
    
    def create_portfolio_positions(self, template_name: str) -> Optional[Dict[str, float]]:
        """
        Create portfolio positions dictionary for use with data loaders
        
        Returns:
            Dictionary mapping option keys to position sizes, or None if template not found
        """
        template = self.get_portfolio_template(template_name)
        if not template:
            return None
        
        return template.positions.copy()
    
    def validate_portfolio_data(self, template_name: str) -> Dict[str, Any]:
        """
        Validate that required data is available for a portfolio template
        
        Returns:
            Validation results with availability information and recommendations
        """
        template = self.get_portfolio_template(template_name)
        if not template:
            return {
                'valid': False,
                'error': f'Portfolio template "{template_name}" not found',
                'available_templates': self.list_available_templates()
            }
        
        availability = self._check_data_availability(template)
        
        # Determine if portfolio is valid for training
        valid = availability['available_positions'] > 0
        
        validation_result = {
            'valid': valid,
            'template_name': template_name,
            'availability': availability,
            'recommendations': []
        }
        
        # Add recommendations based on availability
        if availability['available_positions'] == 0:
            validation_result['recommendations'].append(
                "No data available for any positions. Run preprocessing first."
            )
        elif availability['available_positions'] < availability['total_positions']:
            validation_result['recommendations'].append(
                f"Only {availability['available_positions']}/{availability['total_positions']} positions have data. "
                f"Consider running preprocessing for missing options."
            )
        
        if availability['missing_data']:
            validation_result['recommendations'].append(
                f"Missing data for: {', '.join(availability['missing_data'])}"
            )
        
        return validation_result
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of all available portfolios"""
        summary = {
            'total_templates': len(self.templates),
            'templates': {}
        }
        
        for name in self.templates.keys():
            info = self.get_template_info(name)
            if info:
                summary['templates'][name] = {
                    'description': info['description'],
                    'positions': len(info['positions']),
                    'data_available': info['data_availability']['available_positions'],
                    'availability_pct': info['data_availability']['availability_percentage']
                }
        
        return summary


# Convenience functions for backward compatibility

def create_portfolio_manager(config: Config) -> PortfolioManager:
    """Create portfolio manager instance"""
    return PortfolioManager(config)


def load_portfolio_template(template_name: str, config: Config) -> Optional[Dict[str, float]]:
    """Load portfolio positions from template - convenience function"""
    manager = PortfolioManager(config)
    return manager.create_portfolio_positions(template_name)