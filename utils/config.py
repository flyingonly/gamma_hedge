"""
Unified configuration management using the enhanced architecture
Supports multiple configuration sources, validation, and hot reload
"""

from typing import Dict, Any, Optional
from common.enhanced_config_manager import (
    EnhancedConfigManager,
    UnifiedConfigProvider, 
    get_global_config, 
    get_global_manager,
    create_default_config_provider
)
from common.interfaces import BaseConfig
from common.config_sources import create_default_config_sources
from common.config_validator import create_default_validator

# Initialize enhanced config provider
_config_provider = create_default_config_provider()

def initialize_all_configs():
    """
    Initialize all configuration types by importing their modules
    This breaks circular dependencies by using lazy loading
    """
    try:
        # Import modules to register their config types
        import data.config  # Registers 'market' config
        import training.config  # Registers 'training', 'model', 'delta_hedge' configs
        
        # Try to import delta_hedge config if available
        try:
            import delta_hedge.config  # May register additional delta configs
        except ImportError:
            pass  # Delta hedge not available, skip
            
    except ImportError as e:
        # Handle missing modules gracefully
        print(f"Warning: Could not initialize some configs: {e}")

def get_config() -> Dict[str, BaseConfig]:
    """
    Get default configuration dictionary with validation
    
    Returns:
        Dictionary of validated configuration objects by type
    """
    # Ensure configs are initialized
    initialize_all_configs()
    
    # Create unified config using the provider
    return _config_provider.create_unified_config()

def get_single_config(config_type: str) -> BaseConfig:
    """
    Get a single configuration by type with validation
    
    Args:
        config_type: Type of configuration ('market', 'training', etc.)
        
    Returns:
        Validated configuration object
    """
    initialize_all_configs()
    return get_global_config(config_type)

def update_config(config_type: str, updates: Dict[str, Any], validate: bool = True):
    """
    Update configuration with new values
    
    Args:
        config_type: Type of configuration to update
        updates: Dictionary of field updates
        validate: Whether to validate the updated configuration
    """
    initialize_all_configs()
    manager = get_global_manager()
    manager.update_config(config_type, updates, validate)

def reload_config():
    """
    Reload configuration from all sources
    Useful for picking up changes in config files or environment variables
    """
    manager = get_global_manager()
    manager.reload_config()

def save_config(config_type: str, source_index: int = 1):
    """
    Save configuration to a specific source
    
    Args:
        config_type: Type of configuration to save
        source_index: Index of source to save to (0=default, 1=user config, etc.)
    """
    manager = get_global_manager()
    manager.save_config(config_type, source_index)

def add_config_change_callback(config_type: str, callback):
    """
    Add callback for configuration changes
    
    Args:
        config_type: Type of configuration to watch
        callback: Function called when config changes (old_config, new_config)
    """
    manager = get_global_manager()
    manager.add_change_callback(config_type, callback)

def enable_auto_reload(interval: float = 5.0):
    """
    Enable automatic configuration reloading
    
    Args:
        interval: Reload check interval in seconds
    """
    manager = get_global_manager()
    manager.reload_interval = interval
    manager.start_auto_reload()

def disable_auto_reload():
    """Disable automatic configuration reloading"""
    manager = get_global_manager()
    manager.stop_auto_reload()

def get_config_status() -> Dict[str, Any]:
    """
    Get configuration manager status and diagnostics
    
    Returns:
        Dictionary with status information
    """
    manager = get_global_manager()
    return manager.get_config_status()

def create_unified_config(args=None) -> Dict[str, BaseConfig]:
    """
    Create unified configuration from command line arguments and sources
    
    Args:
        args: Parsed command line arguments (from argparse)
        
    Returns:
        Unified configuration dictionary
    """
    # Get base configuration
    config = get_config()
    
    # Apply command line overrides if provided
    if args:
        config = apply_args_overrides(config, args)
    
    return config

def apply_args_overrides(config: Dict[str, BaseConfig], args) -> Dict[str, BaseConfig]:
    """
    Apply command line argument overrides to configuration
    
    Args:
        config: Base configuration dictionary
        args: Parsed command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    overrides = {}
    
    # Training config overrides
    if 'training' in config:
        training_overrides = {}
        if hasattr(args, 'epochs') and args.epochs is not None:
            training_overrides['n_epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            training_overrides['batch_size'] = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            training_overrides['learning_rate'] = args.learning_rate
            
        if training_overrides:
            overrides['training'] = training_overrides
    
    # Delta hedge config overrides
    if 'delta_hedge' in config:
        delta_overrides = {}
        if hasattr(args, 'delta_hedge') and args.delta_hedge:
            delta_overrides['enable_delta'] = check_delta_availability_safe()
        if hasattr(args, 'underlying_codes') and args.underlying_codes:
            delta_overrides['underlying_codes'] = args.underlying_codes
        if hasattr(args, 'weekly_codes') and args.weekly_codes:
            delta_overrides['weekly_codes'] = args.weekly_codes
        if hasattr(args, 'sequence_length') and args.sequence_length:
            delta_overrides['sequence_length'] = args.sequence_length
            
        if delta_overrides:
            overrides['delta_hedge'] = delta_overrides
    
    # Apply overrides using the enhanced manager
    for config_type, override_values in overrides.items():
        update_config(config_type, override_values)
        config[config_type] = get_single_config(config_type)
    
    return config

def check_delta_availability_safe() -> bool:
    """
    Check delta hedge availability without circular imports
    
    Returns:
        True if delta hedge is available, False otherwise
    """
    try:
        import delta_hedge
        return True
    except ImportError:
        return False

# Legacy compatibility functions
def get_default_config():
    """Legacy compatibility function"""
    return get_config()

def update_config_from_args(config_dict, args):
    """Legacy compatibility function"""
    return apply_args_overrides(config_dict, args)

# Configuration setup helper
def setup_config_system(config_files: Optional[Dict[str, str]] = None,
                       enable_auto_reload: bool = False,
                       reload_interval: float = 5.0) -> UnifiedConfigProvider:
    """
    Setup the configuration system with custom sources
    
    Args:
        config_files: Dictionary of config file paths
        enable_auto_reload: Whether to enable automatic reloading
        reload_interval: Reload check interval in seconds
        
    Returns:
        Configured UnifiedConfigProvider
    """
    from common.config_sources import FileConfigSource, ConfigSourceChain, EnvironmentConfigSource
    
    # Create custom sources if specified
    if config_files:
        sources = []
        for name, path in config_files.items():
            sources.append(FileConfigSource(path))
        sources.append(EnvironmentConfigSource())  # Always include env vars
        source_chain = ConfigSourceChain(sources)
    else:
        source_chain = create_default_config_sources()
    
    # Create enhanced manager with custom settings
    validator = create_default_validator()
    manager = EnhancedConfigManager(
        sources=source_chain,
        validator=validator,
        auto_reload=enable_auto_reload,
        reload_interval=reload_interval
    )
    
    # Create and return provider
    return UnifiedConfigProvider(manager)