"""
Unified Configuration Management - Migrated to New Architecture

This module provides backward compatibility wrappers for the old configuration
system while using the new unified configuration architecture.

MIGRATION NOTICE: This file has been updated to use the new core.config system.
All functionality now leverages the simplified, unified configuration approach.
"""

import warnings
from typing import Dict, Any, Optional

# Import new unified configuration system
from core.config import Config, create_config

# Deprecated: Issue deprecation warning for old usage patterns
def _deprecated_warning(old_function: str):
    warnings.warn(
        f"{old_function} is deprecated. Use core.config.create_config() instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Global configuration instance (for backward compatibility)
_global_config: Optional[Config] = None


def initialize_all_configs():
    """
    Initialize all configuration types - DEPRECATED
    
    The new unified configuration system automatically initializes all config types.
    This function is maintained for backward compatibility only.
    """
    _deprecated_warning("initialize_all_configs")
    
    global _global_config
    if _global_config is None:
        _global_config = create_config()


def get_config(config_type: str = None) -> Any:
    """
    Get configuration by type - BACKWARD COMPATIBILITY WRAPPER
    
    Args:
        config_type: Configuration type ('market', 'training', etc.)
        
    Returns:
        Configuration object or unified config if no type specified
    """
    global _global_config
    if _global_config is None:
        _global_config = create_config()
    
    if config_type is None:
        # Return entire config object
        return _global_config
    
    # Return specific config section
    if hasattr(_global_config, config_type):
        return getattr(_global_config, config_type)
    else:
        # Try dictionary-style access for backward compatibility
        return _global_config[config_type]


def create_unified_config(args=None, **kwargs) -> Config:
    """
    Create unified configuration - BACKWARD COMPATIBILITY WRAPPER
    
    This function maintains backward compatibility with the old API
    while using the new unified configuration system.
    
    Args:
        args: Command line arguments (argparse namespace or dict)
        **kwargs: Additional configuration overrides
        
    Returns:
        Unified configuration object
    """
    # Convert args to dictionary if needed
    if args is not None:
        if hasattr(args, '__dict__'):
            # argparse.Namespace object
            cli_args = vars(args)
        elif isinstance(args, dict):
            # Already a dictionary
            cli_args = args
        else:
            # Try to convert to dict
            cli_args = dict(args) if args else {}
    else:
        cli_args = {}
    
    # Merge with additional kwargs
    cli_args.update(kwargs)
    
    # Create configuration using new system
    return create_config(cli_args=cli_args)


def get_global_config(config_type: str = None) -> Any:
    """
    Get global configuration instance - BACKWARD COMPATIBILITY WRAPPER
    
    Args:
        config_type: Configuration type to retrieve
        
    Returns:
        Configuration object
    """
    return get_config(config_type)


def get_global_manager():
    """
    Get global configuration manager - DEPRECATED
    
    The new system doesn't use separate managers.
    Returns the global config for compatibility.
    """
    _deprecated_warning("get_global_manager")
    return get_config()


# Backward compatibility aliases
def create_default_config_provider():
    """DEPRECATED: Use core.config.create_config() instead"""
    _deprecated_warning("create_default_config_provider")
    return get_config()


# Initialize global config on import for backward compatibility
initialize_all_configs()


# Public API for backward compatibility
__all__ = [
    'get_config',
    'create_unified_config', 
    'get_global_config',
    'initialize_all_configs'
]