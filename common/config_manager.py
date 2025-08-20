"""
Centralized configuration manager to resolve circular dependencies
"""

from typing import Dict, Any, Type, Optional
from .interfaces import BaseConfig, ConfigProvider
from .exceptions import ConfigurationError

class ConfigRegistry:
    """Global configuration registry"""
    
    def __init__(self):
        self._configs: Dict[str, BaseConfig] = {}
        self._config_types: Dict[str, Type[BaseConfig]] = {}
    
    def register_config_type(self, name: str, config_class: Type[BaseConfig]):
        """Register a configuration type"""
        self._config_types[name] = config_class
    
    def create_config(self, name: str, **kwargs) -> BaseConfig:
        """Create configuration instance"""
        if name not in self._config_types:
            raise ConfigurationError(f"Unknown config type: {name}")
        
        config_class = self._config_types[name]
        return config_class(**kwargs)
    
    def set_config(self, name: str, config: BaseConfig):
        """Set configuration instance"""
        self._configs[name] = config
    
    def get_config(self, name: str) -> BaseConfig:
        """Get configuration instance"""
        if name not in self._configs:
            # Try to create default config if type is registered
            if name in self._config_types:
                self._configs[name] = self._config_types[name]()
            else:
                raise ConfigurationError(f"Config '{name}' not found and no default available")
        
        return self._configs[name]
    
    def update_config(self, name: str, updates: Dict[str, Any]):
        """Update configuration with new values"""
        config = self.get_config(name)
        
        # Convert to dict, update, and recreate
        config_dict = config.to_dict()
        config_dict.update(updates)
        
        # Validate and set updated config
        new_config = self._config_types[name].from_dict(config_dict)
        if not new_config.validate():
            raise ConfigurationError(f"Updated config for '{name}' failed validation")
        
        self._configs[name] = new_config
    
    def list_configs(self) -> list[str]:
        """List available configurations"""
        return list(self._configs.keys())
    
    def list_config_types(self) -> list[str]:
        """List registered configuration types"""
        return list(self._config_types.keys())

# Global registry instance
_global_registry = ConfigRegistry()

class UnifiedConfigProvider(ConfigProvider):
    """Unified configuration provider using global registry"""
    
    def __init__(self, registry: Optional[ConfigRegistry] = None):
        self.registry = registry or _global_registry
    
    def get_config(self, config_type: str) -> BaseConfig:
        """Get configuration by type"""
        return self.registry.get_config(config_type)
    
    def register_config(self, config_type: str, config: BaseConfig) -> None:
        """Register configuration"""
        self.registry.set_config(config_type, config)
    
    def register_config_type(self, config_type: str, config_class: Type[BaseConfig]):
        """Register configuration type"""
        self.registry.register_config_type(config_type, config_class)
    
    def create_unified_config(self, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, BaseConfig]:
        """Create unified configuration dictionary"""
        overrides = overrides or {}
        
        unified = {}
        for config_name in self.registry.list_config_types():
            try:
                config = self.registry.get_config(config_name)
                
                # Apply overrides if provided
                if config_name in overrides:
                    config_dict = config.to_dict()
                    config_dict.update(overrides[config_name])
                    config = type(config).from_dict(config_dict)
                
                unified[config_name] = config
            except ConfigurationError:
                # Skip configs that can't be created
                continue
        
        return unified

# Convenience functions using global registry
def register_config_type(name: str, config_class: Type[BaseConfig]):
    """Register a configuration type globally"""
    _global_registry.register_config_type(name, config_class)

def get_global_config(name: str) -> BaseConfig:
    """Get configuration from global registry"""
    return _global_registry.get_config(name)

def set_global_config(name: str, config: BaseConfig):
    """Set configuration in global registry"""
    _global_registry.set_config(name, config)

def create_default_config_provider() -> UnifiedConfigProvider:
    """Create default config provider"""
    return UnifiedConfigProvider()

# Initialize default configurations
def initialize_default_configs():
    """Initialize default configuration types"""
    # This will be called by each module to register its config types
    # Breaking the circular dependency by not importing here
    pass