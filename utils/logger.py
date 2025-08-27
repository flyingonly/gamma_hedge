"""
Unified Logging System for Gamma Hedge Project

This module provides a centralized logging system that replaces the fragmented
logging approach across the project. It offers:
- Configurable log levels with global threshold filtering
- Thread-safe singleton pattern
- Consistent formatting across all modules
- File and console output support
- Integration with the unified configuration system

Usage:
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("This is an info message")
    logger.error("This is an error message")
    
Log Levels (following Python logging convention):
    DEBUG = 10    # Detailed diagnostic information
    INFO = 20     # General information about program execution  
    WARNING = 30  # Warning about potential issues
    ERROR = 40    # Error conditions that don't stop execution
    CRITICAL = 50 # Critical errors that may stop execution
"""

import logging
import threading
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from enum import IntEnum


class LogLevel(IntEnum):
    """Standard logging levels"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class UnifiedLogger:
    """
    Thread-safe singleton logger with configurable output levels.
    
    This class provides a unified interface for all logging operations
    across the project, with centralized configuration and consistent formatting.
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            # Default configuration
            self._log_level = LogLevel.INFO
            self._log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            self._log_file = None
            self._loggers = {}  # Cache of module loggers
            
            # Setup root logger
            self._setup_root_logger()
            self._initialized = True
    
    def _setup_root_logger(self):
        """Configure the root logger with consistent formatting"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Allow all levels, filter at handler level
        
        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._log_level)
        console_formatter = logging.Formatter(self._log_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self._log_file:
            self._add_file_handler()
    
    def _add_file_handler(self):
        """Add file handler if log file is specified"""
        try:
            # Ensure log directory exists
            log_path = Path(self._log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self._log_file)
            file_handler.setLevel(self._log_level)
            file_formatter = logging.Formatter(self._log_format)
            file_handler.setFormatter(file_formatter)
            
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console logging if file setup fails
            console_logger = logging.getLogger(__name__)
            console_logger.warning(f"Failed to setup file logging: {e}")
    
    def configure(self, 
                  log_level: Optional[str] = None,
                  log_file: Optional[str] = None,
                  log_format: Optional[str] = None):
        """
        Configure the logger with new parameters.
        
        Args:
            log_level: Minimum log level to output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file
            log_format: Custom log format string
        """
        with self._lock:
            config_changed = False
            
            # Update log level
            if log_level is not None:
                if isinstance(log_level, str):
                    try:
                        new_level = LogLevel[log_level.upper()]
                        if new_level != self._log_level:
                            self._log_level = new_level
                            config_changed = True
                    except KeyError:
                        raise ValueError(f"Invalid log level: {log_level}. "
                                       f"Valid levels: {list(LogLevel.__members__.keys())}")
                elif isinstance(log_level, (int, LogLevel)):
                    if log_level != self._log_level:
                        self._log_level = LogLevel(log_level)
                        config_changed = True
            
            # Update log file
            if log_file is not None and log_file != self._log_file:
                self._log_file = log_file
                config_changed = True
            
            # Update log format
            if log_format is not None and log_format != self._log_format:
                self._log_format = log_format
                config_changed = True
            
            # Reconfigure if anything changed
            if config_changed:
                self._setup_root_logger()
                
                # Update existing handlers' levels
                root_logger = logging.getLogger()
                for handler in root_logger.handlers:
                    handler.setLevel(self._log_level)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a logger instance for a specific module.
        
        Args:
            name: Logger name, typically __name__ of the calling module
            
        Returns:
            Configured logger instance
        """
        if name is None:
            name = 'gamma_hedge'
        
        # Return cached logger if exists
        if name in self._loggers:
            return self._loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Allow all levels, filtering at handler level
        
        # Cache the logger
        self._loggers[name] = logger
        return logger
    
    def should_log(self, level: int) -> bool:
        """
        Check if a message at the given level should be logged.
        
        This method provides early exit for performance optimization,
        avoiding expensive string formatting for filtered messages.
        
        Args:
            level: Log level to check
            
        Returns:
            True if the level should be logged, False otherwise
        """
        return level >= self._log_level
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current logger configuration"""
        return {
            'log_level': self._log_level.name,
            'log_level_value': int(self._log_level),
            'log_file': self._log_file,
            'log_format': self._log_format,
            'active_loggers': len(self._loggers)
        }


# Global logger instance
_unified_logger = UnifiedLogger()


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    This is the main entry point for getting logger instances throughout
    the project. It provides a consistent interface while maintaining
    the familiar logging API.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        Configured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Starting application")
        logger.error("Error occurred: %s", error_msg)
    """
    return _unified_logger.get_logger(name)


def configure_logging(log_level: Optional[str] = None,
                     log_file: Optional[str] = None, 
                     log_format: Optional[str] = None):
    """
    Configure the global logging system.
    
    This function should be called early in application startup to
    configure logging behavior globally.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Custom log format string
        
    Example:
        configure_logging(log_level='DEBUG', log_file='logs/app.log')
    """
    _unified_logger.configure(log_level, log_file, log_format)


def should_log(level: int) -> bool:
    """
    Check if logging at the specified level is enabled.
    
    Use this for performance optimization when log message generation
    is expensive.
    
    Args:
        level: Log level to check (use LogLevel enum)
        
    Returns:
        True if logging is enabled for this level
        
    Example:
        if should_log(LogLevel.DEBUG):
            expensive_debug_info = generate_debug_data()
            logger.debug("Debug data: %s", expensive_debug_info)
    """
    return _unified_logger.should_log(level)


def get_logging_config() -> Dict[str, Any]:
    """
    Get current logging configuration.
    
    Returns:
        Dictionary with current logging settings
    """
    return _unified_logger.get_current_config()


def set_log_level(level: str):
    """
    Convenience function to set log level.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    configure_logging(log_level=level)


# Export key classes and functions
__all__ = [
    'LogLevel',
    'get_logger', 
    'configure_logging',
    'should_log',
    'get_logging_config',
    'set_log_level'
]