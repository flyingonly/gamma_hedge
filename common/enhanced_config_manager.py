"""
Enhanced configuration manager with multi-source support, validation, and hot reload
"""

import threading
import time
from typing import Dict, Any, Type, Optional, List, Callable
from pathlib import Path
from .interfaces import BaseConfig, ConfigProvider
from .exceptions import ConfigurationError
from .config_sources import BaseConfigSource, ConfigSourceChain, create_default_config_sources
from .config_validator import ConfigValidator, create_default_validator

class EnhancedConfigManager:
    """Enhanced configuration manager with advanced features"""
    
    def __init__(self, 
                 sources: Optional[ConfigSourceChain] = None,
                 validator: Optional[ConfigValidator] = None,
                 auto_reload: bool = False,
                 reload_interval: float = 5.0):
        
        self.sources = sources or create_default_config_sources()
        self.validator = validator or create_default_validator()
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        
        # Configuration storage
        self._configs: Dict[str, BaseConfig] = {}
        self._config_types: Dict[str, Type[BaseConfig]] = {}
        self._raw_config: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._reload_thread: Optional[threading.Thread] = None
        self._stop_reload = threading.Event()
        
        # Change callbacks
        self._change_callbacks: Dict[str, List[Callable]] = {}
        
        # Initialize but don't load configuration immediately to avoid circular imports
        self._config_loaded = False
        
        # Start auto-reload if enabled
        if self.auto_reload:
            self.start_auto_reload()
    
    def register_config_type(self, name: str, config_class: Type[BaseConfig]):
        """Register a configuration type"""
        with self._lock:
            self._config_types[name] = config_class
            
            # If we have raw config for this type, create the instance
            if name in self._raw_config:
                self._create_config_instance(name)
    
    def get_config(self, name: str) -> BaseConfig:
        """Get configuration instance with validation"""
        with self._lock:
            # Ensure configuration is loaded
            if not self._config_loaded:
                try:
                    self._load_config()
                    self._config_loaded = True
                except Exception as e:
                    # If config loading fails, continue with empty config
                    print(f"Warning: Failed to load configuration files: {e}")
                    self._config_loaded = True
            
            if name not in self._configs:
                if name not in self._config_types:
                    raise ConfigurationError(f"Config type '{name}' not registered")
                
                # Try to create from loaded config first, then fall back to default
                if name in self._raw_config:
                    self._create_config_instance(name)
                else:
                    # Create default config
                    self._configs[name] = self._config_types[name]()
                    
                    # Validate default config
                    self._validate_config(name, self._configs[name])
            
            return self._configs[name]
    
    def update_config(self, name: str, updates: Dict[str, Any], validate: bool = True):
        """Update configuration with new values"""
        with self._lock:
            if name not in self._config_types:
                raise ConfigurationError(f"Config type '{name}' not registered")
            
            # Get current config
            current_config = self.get_config(name)
            
            # Merge updates
            config_dict = current_config.to_dict()
            config_dict.update(updates)
            
            # Create new config instance
            new_config = self._config_types[name].from_dict(config_dict)
            
            # Validate if requested
            if validate:
                self._validate_config(name, new_config)
            
            # Store old config for rollback
            old_config = self._configs.get(name)
            
            # Update config
            self._configs[name] = new_config
            self._raw_config[name] = config_dict
            
            # Notify change callbacks
            self._notify_change_callbacks(name, old_config, new_config)
    
    def reload_config(self):
        """Reload configuration from sources"""
        with self._lock:
            old_configs = self._configs.copy()
            
            try:
                # Load raw configuration
                self._load_config()
                
                # Recreate all config instances
                for name in self._config_types:
                    if name in self._raw_config:
                        self._create_config_instance(name)
                
                # Notify change callbacks for all changed configs
                for name, old_config in old_configs.items():
                    new_config = self._configs.get(name)
                    if new_config and old_config.to_dict() != new_config.to_dict():
                        self._notify_change_callbacks(name, old_config, new_config)
                        
            except Exception as e:
                # Rollback on error
                self._configs = old_configs
                raise ConfigurationError(f"Failed to reload configuration: {e}")
    
    def save_config(self, name: str, source_index: int = 1):
        """Save configuration to a specific source"""
        with self._lock:
            if name not in self._configs:
                raise ConfigurationError(f"Config '{name}' not found")
            
            config_dict = self._configs[name].to_dict()
            
            # Try to save to the specified source
            if source_index < len(self.sources.sources):
                source = self.sources.sources[source_index]
                if not source.save({name: config_dict}):
                    raise ConfigurationError(f"Failed to save config to source {source_index}")
            else:
                raise ConfigurationError(f"Invalid source index: {source_index}")
    
    def add_change_callback(self, config_name: str, callback: Callable):
        """Add callback for configuration changes"""
        with self._lock:
            if config_name not in self._change_callbacks:
                self._change_callbacks[config_name] = []
            self._change_callbacks[config_name].append(callback)
    
    def remove_change_callback(self, config_name: str, callback: Callable):
        """Remove configuration change callback"""
        with self._lock:
            if config_name in self._change_callbacks:
                try:
                    self._change_callbacks[config_name].remove(callback)
                except ValueError:
                    pass
    
    def start_auto_reload(self):
        """Start automatic configuration reloading"""
        if self._reload_thread and self._reload_thread.is_alive():
            return
        
        self._stop_reload.clear()
        self._reload_thread = threading.Thread(target=self._auto_reload_worker, daemon=True)
        self._reload_thread.start()
    
    def stop_auto_reload(self):
        """Stop automatic configuration reloading"""
        self._stop_reload.set()
        if self._reload_thread:
            self._reload_thread.join(timeout=1.0)
    
    def get_all_configs(self) -> Dict[str, BaseConfig]:
        """Get all loaded configurations"""
        with self._lock:
            return self._configs.copy()
    
    def get_config_status(self) -> Dict[str, Any]:
        """Get configuration manager status"""
        with self._lock:
            return {
                'registered_types': list(self._config_types.keys()),
                'loaded_configs': list(self._configs.keys()),
                'auto_reload': self.auto_reload,
                'reload_interval': self.reload_interval,
                'sources_count': len(self.sources.sources),
            }
    
    def _load_config(self):
        """Load raw configuration from sources"""
        try:
            self._raw_config = self.sources.load()
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from sources: {e}")
    
    def _create_config_instance(self, name: str):
        """Create configuration instance from raw config"""
        if name not in self._config_types:
            return
        
        raw_data = self._raw_config.get(name, {})
        
        try:
            # Create config instance
            config = self._config_types[name].from_dict(raw_data)
            
            # Validate config
            self._validate_config(name, config)
            
            # Store config
            self._configs[name] = config
            
        except Exception as e:
            # Fall back to default config
            print(f"Warning: Failed to create config '{name}' from data: {e}")
            self._configs[name] = self._config_types[name]()
    
    def _validate_config(self, name: str, config: BaseConfig):
        """Validate configuration"""
        errors = self.validator.validate(name, config)
        if errors:
            raise ConfigurationError(f"Configuration validation failed for '{name}': {errors}")
    
    def _notify_change_callbacks(self, name: str, old_config: Optional[BaseConfig], new_config: BaseConfig):
        """Notify registered change callbacks"""
        if name in self._change_callbacks:
            for callback in self._change_callbacks[name]:
                try:
                    callback(name, old_config, new_config)
                except Exception as e:
                    print(f"Warning: Config change callback failed: {e}")
    
    def _auto_reload_worker(self):
        """Worker thread for automatic configuration reloading"""
        while not self._stop_reload.wait(self.reload_interval):
            try:
                self.reload_config()
            except Exception as e:
                print(f"Warning: Auto-reload failed: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_auto_reload()

class UnifiedConfigProvider(ConfigProvider):
    """Unified configuration provider using enhanced manager"""
    
    def __init__(self, manager: Optional[EnhancedConfigManager] = None):
        self.manager = manager or EnhancedConfigManager()
    
    def get_config(self, config_type: str) -> BaseConfig:
        """Get configuration by type"""
        return self.manager.get_config(config_type)
    
    def register_config(self, config_type: str, config: BaseConfig) -> None:
        """Register a configuration"""
        # Update the manager's stored config directly
        with self.manager._lock:
            self.manager._configs[config_type] = config
            # Also store the raw config data for consistency
            self.manager._raw_config[config_type] = config.to_dict()
    
    def create_unified_config(self, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, BaseConfig]:
        """Create unified configuration dictionary"""
        configs = self.manager.get_all_configs()
        
        # Apply overrides if provided
        if overrides:
            for config_name, override_values in overrides.items():
                if config_name in configs:
                    self.manager.update_config(config_name, override_values)
                    configs[config_name] = self.manager.get_config(config_name)
        
        return configs

# Global instances
_global_manager: Optional[EnhancedConfigManager] = None
_global_provider: Optional[UnifiedConfigProvider] = None

def get_global_manager() -> EnhancedConfigManager:
    """Get global configuration manager"""
    global _global_manager
    if _global_manager is None:
        _global_manager = EnhancedConfigManager()
    return _global_manager

def get_global_provider() -> UnifiedConfigProvider:
    """Get global configuration provider"""
    global _global_provider
    if _global_provider is None:
        _global_provider = UnifiedConfigProvider(get_global_manager())
    return _global_provider

def register_config_type(name: str, config_class: Type[BaseConfig]):
    """Register configuration type globally"""
    get_global_manager().register_config_type(name, config_class)

def get_global_config(config_type: str) -> BaseConfig:
    """Get global configuration by type"""
    return get_global_manager().get_config(config_type)

def create_default_config_provider() -> UnifiedConfigProvider:
    """Create default configuration provider"""
    return get_global_provider()