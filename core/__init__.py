"""
Core module for Gamma Hedge system

This module provides the foundational components for the modular architecture:
- Unified configuration management
- Standard data exchange interfaces  
- Application orchestration
"""

from .config import Config, create_config
from .interfaces import TrainingData, DataProvider, TrainingPhase
from .orchestrator import ApplicationOrchestrator, WorkflowType, create_orchestrator

__all__ = [
    # Configuration system
    'Config', 
    'create_config',
    
    # Data interfaces
    'TrainingData', 
    'DataProvider', 
    'TrainingPhase',
    
    # Application orchestration
    'ApplicationOrchestrator',
    'WorkflowType',
    'create_orchestrator'
]