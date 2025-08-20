"""
Custom exceptions for gamma hedge system
"""

class GammaHedgeException(Exception):
    """Base exception for gamma hedge system"""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message

class ConfigurationError(GammaHedgeException):
    """Configuration related errors"""
    pass

class DataProviderError(GammaHedgeException):
    """Data provider related errors"""
    pass

class ModelError(GammaHedgeException):
    """Model related errors"""
    pass

class TrainingError(GammaHedgeException):
    """Training related errors"""
    pass

class DependencyError(GammaHedgeException):
    """Dependency related errors (e.g., optional modules not available)"""
    pass

class ValidationError(GammaHedgeException):
    """Data or parameter validation errors"""
    pass

# Specific error types for better error handling
class InvalidShapeError(ValidationError):
    """Tensor shape validation error"""
    pass

class MissingDependencyError(DependencyError):
    """Required dependency not found"""
    pass

class IncompatibleConfigError(ConfigurationError):
    """Configuration incompatible with provider/model"""
    pass