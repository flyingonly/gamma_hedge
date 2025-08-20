"""
Configuration validation and schema checking
"""

from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
from .interfaces import BaseConfig
from .exceptions import ConfigurationError

@dataclass
class ValidationRule:
    """Single validation rule for a configuration field"""
    field_name: str
    required: bool = False
    data_type: Optional[Type] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    validator_func: Optional[callable] = None
    error_message: Optional[str] = None

class ConfigValidator:
    """Configuration validator with flexible rules"""
    
    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}
    
    def add_rule(self, config_type: str, rule: ValidationRule):
        """Add validation rule for a specific config type"""
        if config_type not in self.rules:
            self.rules[config_type] = []
        self.rules[config_type].append(rule)
    
    def add_rules(self, config_type: str, rules: List[ValidationRule]):
        """Add multiple validation rules"""
        for rule in rules:
            self.add_rule(config_type, rule)
    
    def validate(self, config_type: str, config: BaseConfig) -> List[str]:
        """Validate configuration and return list of error messages"""
        errors = []
        
        if config_type not in self.rules:
            return errors  # No rules, no validation
        
        config_dict = config.to_dict()
        
        for rule in self.rules[config_type]:
            error = self._validate_field(config_dict, rule)
            if error:
                errors.append(error)
        
        return errors
    
    def validate_strict(self, config_type: str, config: BaseConfig):
        """Validate configuration and raise exception on first error"""
        errors = self.validate(config_type, config)
        if errors:
            raise ConfigurationError(f"Configuration validation failed for {config_type}: {errors[0]}")
    
    def _validate_field(self, config_dict: Dict[str, Any], rule: ValidationRule) -> Optional[str]:
        """Validate a single field according to rule"""
        field_name = rule.field_name
        value = config_dict.get(field_name)
        
        # Check required fields
        if rule.required and (value is None):
            return rule.error_message or f"Required field '{field_name}' is missing"
        
        # Skip validation for None values if not required
        if value is None:
            return None
        
        # Type checking
        if rule.data_type and not isinstance(value, rule.data_type):
            return rule.error_message or f"Field '{field_name}' must be of type {rule.data_type.__name__}, got {type(value).__name__}"
        
        # Range checking for numeric values
        if rule.min_value is not None and isinstance(value, (int, float)):
            if value < rule.min_value:
                return rule.error_message or f"Field '{field_name}' must be >= {rule.min_value}, got {value}"
        
        if rule.max_value is not None and isinstance(value, (int, float)):
            if value > rule.max_value:
                return rule.error_message or f"Field '{field_name}' must be <= {rule.max_value}, got {value}"
        
        # Allowed values checking
        if rule.allowed_values and value not in rule.allowed_values:
            return rule.error_message or f"Field '{field_name}' must be one of {rule.allowed_values}, got {value}"
        
        # Custom validator function
        if rule.validator_func:
            try:
                if not rule.validator_func(value):
                    return rule.error_message or f"Field '{field_name}' failed custom validation"
            except Exception as e:
                return rule.error_message or f"Field '{field_name}' validation error: {e}"
        
        return None

def create_default_validator() -> ConfigValidator:
    """Create validator with default rules for all config types"""
    validator = ConfigValidator()
    
    # Market config validation rules
    market_rules = [
        ValidationRule('n_assets', required=True, data_type=int, min_value=1, max_value=100),
        ValidationRule('n_timesteps', required=True, data_type=int, min_value=1, max_value=10000),
        ValidationRule('volatility', required=True, data_type=float, min_value=0.0, max_value=5.0),
        ValidationRule('dt', required=True, data_type=float, min_value=0.0001, max_value=1.0),
        ValidationRule('initial_price', required=True, data_type=float, min_value=0.1),
    ]
    validator.add_rules('market', market_rules)
    
    # Training config validation rules
    training_rules = [
        ValidationRule('batch_size', required=True, data_type=int, min_value=1, max_value=1024),
        ValidationRule('learning_rate', required=True, data_type=float, min_value=1e-6, max_value=1.0),
        ValidationRule('n_epochs', required=True, data_type=int, min_value=1, max_value=10000),
        ValidationRule('device', required=True, data_type=str, allowed_values=['cpu', 'cuda']),
        ValidationRule('checkpoint_interval', required=True, data_type=int, min_value=1),
    ]
    validator.add_rules('training', training_rules)
    
    # Model config validation rules
    model_rules = [
        ValidationRule('dropout_rate', required=True, data_type=float, min_value=0.0, max_value=1.0),
        ValidationRule('hidden_dims', required=True, data_type=list, 
                      validator_func=lambda x: all(isinstance(dim, int) and dim > 0 for dim in x)),
    ]
    validator.add_rules('model', model_rules)
    
    # Delta hedge config validation rules
    delta_rules = [
        ValidationRule('enable_delta', required=True, data_type=bool),
        ValidationRule('sequence_length', required=True, data_type=int, min_value=1, max_value=10000),
        ValidationRule('underlying_codes', required=True, data_type=list,
                      validator_func=lambda x: len(x) > 0 and all(isinstance(code, str) for code in x)),
        ValidationRule('weekly_codes', required=True, data_type=list,
                      validator_func=lambda x: len(x) > 0 and all(isinstance(code, str) for code in x)),
    ]
    validator.add_rules('delta_hedge', delta_rules)
    
    return validator