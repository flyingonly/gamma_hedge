"""
Application Orchestrator for Gamma Hedge System

This module provides the central orchestration layer that coordinates
all application workflows, replacing the complex logic scattered across
main.py and run_production_training.py with a clean, unified approach.

Key responsibilities:
- Workflow coordination and execution
- Error handling and recovery
- Resource management
- Logging and monitoring
- Integration between all system components
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum

import torch
import numpy as np

from core.config import Config
from core.interfaces import TrainingData, DataProvider, TrainingPhase
from utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowType(Enum):
    """Available workflow types"""
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREPROCESSING = "preprocessing"
    VALIDATION = "validation"


class ExecutionStatus(Enum):
    """Execution status tracking"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowResult:
    """Results from workflow execution"""
    
    def __init__(self, 
                 workflow_type: WorkflowType,
                 status: ExecutionStatus,
                 results: Optional[Dict[str, Any]] = None,
                 error: Optional[Exception] = None,
                 execution_time: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        self.workflow_type = workflow_type
        self.status = status
        self.results = results or {}
        self.error = error
        self.execution_time = execution_time
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    @property
    def succeeded(self) -> bool:
        return self.status == ExecutionStatus.COMPLETED
    
    @property
    def failed(self) -> bool:
        return self.status == ExecutionStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'workflow_type': self.workflow_type.value,
            'status': self.status.value,
            'results': self.results,
            'error': str(self.error) if self.error else None,
            'execution_time': self.execution_time,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class ApplicationOrchestrator:
    """
    Central orchestrator for all application workflows
    
    This class coordinates the execution of training, evaluation, and
    preprocessing workflows, providing a unified interface and consistent
    error handling across the entire system.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.execution_history: List[WorkflowResult] = []
        self._setup_logging()
        self._setup_device()
        self._setup_reproducibility()
        
        # Initialize core components
        from data import create_unified_data_provider, PortfolioManager
        self.data_provider = create_unified_data_provider(config)
        self.portfolio_manager = PortfolioManager(config) if config.delta_hedge.enable_delta else None
        
        # Workflow callbacks
        self._progress_callbacks: Dict[WorkflowType, List[Callable]] = {}
        
        logger.info("Application orchestrator initialized")
    
    def _setup_logging(self):
        """Setup application-wide logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_device(self):
        """Setup compute device"""
        self.device = torch.device(
            self.config.training.device if torch.cuda.is_available() and self.config.training.device == 'cuda'
            else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def _setup_reproducibility(self):
        """Setup reproducible random seeds"""
        seed = 42  # Fixed seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Make operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"Reproducibility setup with seed: {seed}")
    
    def register_progress_callback(self, workflow_type: WorkflowType, callback: Callable):
        """Register callback for workflow progress updates"""
        if workflow_type not in self._progress_callbacks:
            self._progress_callbacks[workflow_type] = []
        self._progress_callbacks[workflow_type].append(callback)
    
    def _notify_progress(self, workflow_type: WorkflowType, progress: Dict[str, Any]):
        """Notify registered progress callbacks"""
        if workflow_type in self._progress_callbacks:
            for callback in self._progress_callbacks[workflow_type]:
                try:
                    callback(progress)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
    
    def execute_workflow(self, workflow_type: WorkflowType, **kwargs) -> WorkflowResult:
        """
        Execute a workflow with comprehensive error handling
        
        Args:
            workflow_type: Type of workflow to execute
            **kwargs: Workflow-specific arguments
            
        Returns:
            WorkflowResult with execution details
        """
        start_time = time.time()
        
        logger.info(f"Starting {workflow_type.value} workflow")
        
        try:
            # Execute appropriate workflow
            if workflow_type == WorkflowType.TRAINING:
                results = self._execute_training_workflow(**kwargs)
            elif workflow_type == WorkflowType.EVALUATION:
                results = self._execute_evaluation_workflow(**kwargs)
            elif workflow_type == WorkflowType.PREPROCESSING:
                results = self._execute_preprocessing_workflow(**kwargs)
            elif workflow_type == WorkflowType.VALIDATION:
                results = self._execute_validation_workflow(**kwargs)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            execution_time = time.time() - start_time
            
            # Create successful result
            result = WorkflowResult(
                workflow_type=workflow_type,
                status=ExecutionStatus.COMPLETED,
                results=results,
                execution_time=execution_time,
                metadata={'config_summary': self._get_config_summary()}
            )
            
            logger.info(f"Completed {workflow_type.value} workflow in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create failed result
            result = WorkflowResult(
                workflow_type=workflow_type,
                status=ExecutionStatus.FAILED,
                error=e,
                execution_time=execution_time,
                metadata={'traceback': traceback.format_exc()}
            )
            
            logger.error(f"Failed {workflow_type.value} workflow: {e}")
            logger.debug(traceback.format_exc())
        
        # Store in execution history
        self.execution_history.append(result)
        
        return result
    
    def _execute_training_workflow(self, **kwargs) -> Dict[str, Any]:
        """Execute training workflow"""
        from training.engine import UnifiedTrainingEngine
        
        # Extract training parameters
        n_epochs = kwargs.get('n_epochs', self.config.training.n_epochs)
        batch_size = kwargs.get('batch_size', self.config.training.batch_size)
        portfolio_name = kwargs.get('portfolio', None)
        resume = kwargs.get('resume', False)
        checkpoint_interval = kwargs.get('checkpoint_interval', 10)
        
        # Setup portfolio if specified
        if portfolio_name and self.portfolio_manager:
            portfolio_positions = self.portfolio_manager.create_portfolio_positions(portfolio_name)
            if not portfolio_positions:
                raise ValueError(f"Portfolio template '{portfolio_name}' not found")
            
            # Update data provider with portfolio
            from data import create_unified_data_provider
            self.data_provider = create_unified_data_provider(self.config, data_source="precomputed_greeks")
        
        # Create training engine
        engine = UnifiedTrainingEngine(
            config=self.config,
            data_provider=self.data_provider,
            device=self.device
        )
        
        # Setup progress callback
        def progress_callback(progress_info):
            self._notify_progress(WorkflowType.TRAINING, progress_info)
        
        # Execute training
        training_results = engine.train(
            n_epochs=n_epochs,
            batch_size=batch_size,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            progress_callback=progress_callback
        )
        
        return {
            'training_history': training_results['history'],
            'best_epoch': training_results.get('best_epoch'),
            'final_metrics': training_results.get('final_metrics'),
            'model_path': training_results.get('model_path'),
            'checkpoints': training_results.get('checkpoints', [])
        }
    
    def _execute_evaluation_workflow(self, **kwargs) -> Dict[str, Any]:
        """Execute evaluation workflow"""
        from training.engine import UnifiedTrainingEngine
        
        model_path = kwargs.get('model_path', 'checkpoints/best_model.pth')
        eval_samples = kwargs.get('eval_samples', 1000)
        
        # Create training engine for evaluation
        engine = UnifiedTrainingEngine(
            config=self.config,
            data_provider=self.data_provider,
            device=self.device
        )
        
        # Load model if specified
        if model_path and Path(model_path).exists():
            engine.load_checkpoint(model_path)
        
        # Execute evaluation
        eval_results = engine.evaluate(
            n_samples=eval_samples,
            batch_size=self.config.training.batch_size
        )
        
        return {
            'evaluation_metrics': eval_results,
            'model_path': model_path,
            'eval_samples': eval_samples
        }
    
    def _execute_preprocessing_workflow(self, **kwargs) -> Dict[str, Any]:
        """Execute preprocessing workflow"""
        from data.greeks_preprocessor import GreeksPreprocessor
        
        # Extract preprocessing parameters
        weekly_codes = kwargs.get('weekly_codes', self.config.delta_hedge.weekly_codes)
        force_reprocess = kwargs.get('force_reprocess', self.config.greeks.force_reprocess)
        mode = kwargs.get('mode', self.config.greeks.preprocessing_mode)
        
        # Create preprocessor
        preprocessor = GreeksPreprocessor(self.config)
        
        # Execute preprocessing
        results = preprocessor.preprocess_batch(
            weekly_codes=weekly_codes,
            force_reprocess=force_reprocess,
            mode=mode
        )
        
        return {
            'processed_options': results.get('processed', []),
            'failed_options': results.get('failed', []),
            'statistics': results.get('statistics', {}),
            'preprocessing_mode': mode
        }
    
    def _execute_validation_workflow(self, **kwargs) -> Dict[str, Any]:
        """Execute validation workflow"""
        validation_type = kwargs.get('validation_type', 'system')
        
        results = {
            'validation_type': validation_type,
            'checks': {}
        }
        
        if validation_type in ['system', 'all']:
            # System validation checks
            results['checks']['config_validation'] = self._validate_configuration()
            results['checks']['data_availability'] = self._validate_data_availability()
            results['checks']['model_compatibility'] = self._validate_model_compatibility()
        
        if validation_type in ['data', 'all']:
            # Data-specific validation
            results['checks']['data_quality'] = self._validate_data_quality()
        
        return results
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration"""
        try:
            self.config.validate()
            return {'status': 'valid', 'issues': []}
        except Exception as e:
            return {'status': 'invalid', 'issues': [str(e)]}
    
    def _validate_data_availability(self) -> Dict[str, Any]:
        """Validate data availability"""
        try:
            info = self.data_provider.get_data_info()
            return {'status': 'available', 'info': info}
        except Exception as e:
            return {'status': 'unavailable', 'error': str(e)}
    
    def _validate_model_compatibility(self) -> Dict[str, Any]:
        """Validate model compatibility with configuration"""
        try:
            from models.policy_network import PolicyNetwork
            
            # Calculate input dimensions based on configuration
            if self.config.delta_hedge.enable_delta:
                n_assets = 1  # Delta hedging uses single underlying
            else:
                n_assets = self.config.market.n_assets
            
            # Calculate input dimension using centralized dimension management
            from core.dimensions import calculate_model_input_dim
            input_dim = calculate_model_input_dim(
                n_assets=n_assets,
                include_history=True,
                include_time=True,
                include_delta=True
            )
            
            # Try to create model
            model = PolicyNetwork(
                input_dim=input_dim,
                hidden_dims=self.config.model.hidden_dims,
                dropout_rate=self.config.model.dropout_rate
            )
            
            return {
                'status': 'compatible',
                'input_dim': input_dim,
                'model_params': sum(p.numel() for p in model.parameters())
            }
        except Exception as e:
            return {'status': 'incompatible', 'error': str(e)}
    
    def _validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality"""
        try:
            from data.processors import DataValidator
            
            validator = DataValidator(self.config)
            
            # Get sample data for validation
            sample_data = self.data_provider.get_training_data(
                batch_size=1,
                sequence_length=10,
                phase=TrainingPhase.TRAIN
            )
            
            validation_result = validator.validate_training_data(sample_data)
            
            return validation_result
        except Exception as e:
            return {'status': 'validation_failed', 'error': str(e)}
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for metadata"""
        return {
            'market_assets': self.config.market.n_assets,
            'training_epochs': self.config.training.n_epochs,
            'batch_size': self.config.training.batch_size,
            'delta_hedge_enabled': self.config.delta_hedge.enable_delta,
            'preprocessing_mode': self.config.greeks.preprocessing_mode,
            'device': str(self.device)
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executed workflows"""
        summary = {
            'total_executions': len(self.execution_history),
            'successful_executions': len([r for r in self.execution_history if r.succeeded]),
            'failed_executions': len([r for r in self.execution_history if r.failed]),
            'workflows_by_type': {},
            'recent_executions': []
        }
        
        # Group by workflow type
        for result in self.execution_history:
            workflow_name = result.workflow_type.value
            if workflow_name not in summary['workflows_by_type']:
                summary['workflows_by_type'][workflow_name] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'avg_execution_time': 0.0
                }
            
            summary['workflows_by_type'][workflow_name]['total'] += 1
            if result.succeeded:
                summary['workflows_by_type'][workflow_name]['successful'] += 1
            elif result.failed:
                summary['workflows_by_type'][workflow_name]['failed'] += 1
        
        # Calculate average execution times
        for workflow_type in summary['workflows_by_type']:
            matching_results = [r for r in self.execution_history if r.workflow_type.value == workflow_type]
            if matching_results:
                avg_time = sum(r.execution_time for r in matching_results) / len(matching_results)
                summary['workflows_by_type'][workflow_type]['avg_execution_time'] = avg_time
        
        # Recent executions (last 10)
        recent = sorted(self.execution_history, key=lambda x: x.timestamp, reverse=True)[:10]
        summary['recent_executions'] = [r.to_dict() for r in recent]
        
        return summary
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Orchestrator cleanup completed")


# Convenience functions

def create_orchestrator(config: Config) -> ApplicationOrchestrator:
    """Create application orchestrator instance"""
    return ApplicationOrchestrator(config)


def execute_training(config: Config, **kwargs) -> WorkflowResult:
    """Convenience function for training execution"""
    orchestrator = create_orchestrator(config)
    return orchestrator.execute_workflow(WorkflowType.TRAINING, **kwargs)


def execute_evaluation(config: Config, **kwargs) -> WorkflowResult:
    """Convenience function for evaluation execution"""
    orchestrator = create_orchestrator(config)
    return orchestrator.execute_workflow(WorkflowType.EVALUATION, **kwargs)