"""
Configuration sources and loaders for multiple input formats
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from .exceptions import ConfigurationError

class BaseConfigSource:
    """Base class for configuration sources"""
    
    def load(self) -> Dict[str, Any]:
        """Load configuration data"""
        raise NotImplementedError
    
    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration data"""
        raise NotImplementedError
    
    def exists(self) -> bool:
        """Check if configuration source exists"""
        raise NotImplementedError

class FileConfigSource(BaseConfigSource):
    """Configuration source from JSON/YAML files"""
    
    def __init__(self, file_path: Union[str, Path], format: str = 'auto'):
        self.file_path = Path(file_path)
        self.format = format
        
        if format == 'auto':
            self.format = self._detect_format()
    
    def _detect_format(self) -> str:
        """Detect file format from extension"""
        suffix = self.file_path.suffix.lower()
        if suffix in ['.json']:
            return 'json'
        elif suffix in ['.yaml', '.yml']:
            return 'yaml'
        else:
            return 'json'  # Default
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.exists():
            return {}
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                if self.format == 'json':
                    return json.load(f)
                elif self.format == 'yaml':
                    try:
                        import yaml
                        return yaml.safe_load(f) or {}
                    except ImportError:
                        raise ConfigurationError("PyYAML required for YAML config files")
                else:
                    raise ConfigurationError(f"Unsupported format: {self.format}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {self.file_path}: {e}")
    
    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            # Create directory if it doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                if self.format == 'json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                elif self.format == 'yaml':
                    try:
                        import yaml
                        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
                    except ImportError:
                        raise ConfigurationError("PyYAML required for YAML config files")
                else:
                    raise ConfigurationError(f"Unsupported format: {self.format}")
            return True
        except Exception as e:
            raise ConfigurationError(f"Failed to save config to {self.file_path}: {e}")
    
    def exists(self) -> bool:
        """Check if config file exists"""
        return self.file_path.exists()

class EnvironmentConfigSource(BaseConfigSource):
    """Configuration source from environment variables"""
    
    def __init__(self, prefix: str = 'GAMMA_HEDGE_'):
        self.prefix = prefix.upper()
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.prefix):].lower()
                
                # Try to parse as JSON for complex values
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Treat as string
                    config[config_key] = value
        
        return config
    
    def save(self, config: Dict[str, Any]) -> bool:
        """Environment variables are read-only"""
        return False
    
    def exists(self) -> bool:
        """Check if any environment variables with our prefix exist"""
        return any(key.startswith(self.prefix) for key in os.environ.keys())

class DictConfigSource(BaseConfigSource):
    """Configuration source from Python dictionary"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
    
    def load(self) -> Dict[str, Any]:
        """Return the stored configuration"""
        return self._config.copy()
    
    def save(self, config: Dict[str, Any]) -> bool:
        """Update the stored configuration"""
        self._config = config.copy()
        return True
    
    def exists(self) -> bool:
        """Dictionary source always exists"""
        return True

class ConfigSourceChain:
    """Chain of configuration sources with priority order"""
    
    def __init__(self, sources: List[BaseConfigSource]):
        self.sources = sources
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from all sources, with later sources overriding earlier ones"""
        config = {}
        
        for source in self.sources:
            try:
                source_config = source.load()
                config = self._deep_merge(config, source_config)
            except Exception as e:
                print(f"Warning: Failed to load from config source: {e}")
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

def create_default_config_sources() -> ConfigSourceChain:
    """Create default configuration source chain"""
    sources = []
    
    # Only add file sources if config directory exists
    config_dir = Path('config')
    if config_dir.exists():
        # 1. Default config file (lowest priority)
        default_config = config_dir / 'default.json'
        if default_config.exists():
            sources.append(FileConfigSource(default_config))
        
        # 2. User config file
        user_config = config_dir / 'config.json'
        if user_config.exists():
            sources.append(FileConfigSource(user_config))
        
        # 3. Environment-specific config
        env_name = os.environ.get("ENVIRONMENT", "development")
        env_config = config_dir / f'{env_name}.json'
        if env_config.exists():
            sources.append(FileConfigSource(env_config))
    
    # 4. Environment variables (always available, highest priority)
    sources.append(EnvironmentConfigSource())
    
    return ConfigSourceChain(sources)