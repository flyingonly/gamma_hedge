"""
Unified Training Module for Gamma Hedge

This module provides the unified training interface, consolidating
all training functionality into a clean, simplified system.
"""

# New unified training engine (recommended)
from .engine import UnifiedTrainingEngine, create_training_engine

# Backward compatibility imports
from .trainer import Trainer
from .evaluator import Evaluator

__all__ = [
    # New unified interface (preferred)
    'UnifiedTrainingEngine',
    'create_training_engine',
    
    # Backward compatibility
    'Trainer',
    'Evaluator'
]