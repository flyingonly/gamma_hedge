import time
import traceback
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import torch
from datetime import datetime
from utils.logger import get_logger
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import existing components
from models.policy_network import PolicyNetwork
from training.trainer import Trainer
from training.training_monitor import TrainingMonitor, TrainingMetrics
from core.config import TrainingConfig
from data.option_portfolio_manager import OptionPortfolioManager
from data.data_loader import create_delta_data_loader
from utils.policy_tracker import global_policy_tracker
from utils.policy_visualizer import create_policy_visualizer

class ProductionTrainer:
    """
    Production-grade training orchestrator that coordinates portfolio selection,
    training monitoring, and end-to-end workflow management.
    """
    
    def __init__(self, 
                 portfolio_manager: Optional[OptionPortfolioManager] = None,
                 training_monitor: Optional[TrainingMonitor] = None,
                 device: Optional[str] = None,
                 checkpoint_dir: str = "checkpoints",
                 output_dir: str = "outputs"):
        
        # Initialize components
        self.portfolio_manager = portfolio_manager or OptionPortfolioManager()
        self.training_monitor = training_monitor or TrainingMonitor()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.current_trainer: Optional[Trainer] = None
        self.current_portfolio: Optional[Dict] = None
        self.current_config: Optional[Dict] = None
        
        # Setup logging
        self.logger = get_logger(__name__)
        
        # Alert callbacks
        self._setup_alert_system()
    
    def _setup_alert_system(self):
        """Setup alert system for training monitoring"""
        
        def training_alert_handler(alert_message: str, metrics: TrainingMetrics):
            """Handle training alerts"""
            self.logger.warning(f"TRAINING ALERT: {alert_message}")
            
            # Implement specific alert responses
            if "Loss explosion" in alert_message:
                self.logger.error("Loss explosion detected - consider stopping training")
            elif "stagnation" in alert_message:
                self.logger.warning("Training may have converged or needs learning rate adjustment")
            elif "Slow training" in alert_message:
                self.logger.info("Training is slower than expected - check system resources")
        
        self.training_monitor.add_alert_callback(training_alert_handler)
    
    def select_portfolio(self, config_name: str) -> Dict[str, Any]:
        """
        Select portfolio for training based on configuration
        
        Returns:
            Dictionary with portfolio info and positions
        """
        self.logger.info(f"Selecting portfolio: {config_name}")
        
        try:
            # Get portfolio configuration info
            config_info = self.portfolio_manager.get_config_info(config_name)
            self.logger.info(f"Found {config_info['matching_options']} matching options")
            
            # Select actual portfolio
            portfolio = self.portfolio_manager.select_portfolio(config_name)
            positions = self.portfolio_manager.get_portfolio_positions_dict(portfolio, config_name)
            
            self.current_portfolio = {
                'config_name': config_name,
                'portfolio': portfolio,
                'positions': positions,
                'config_info': config_info
            }
            
            # Log portfolio details
            self.logger.info("Selected portfolio:")
            for position_name, option_info in portfolio.items():
                self.logger.info(f"  {position_name}: {option_info.full_name} "
                               f"(Strike: {option_info.strike}, Type: {option_info.option_type})")
            
            return self.current_portfolio
            
        except Exception as e:
            self.logger.error(f"Portfolio selection failed: {e}")
            raise
    
    def prepare_training_data(self, 
                            batch_size: int = 32,
                            validation_split: float = 0.2,
                            sequence_length: int = 100,
                            training_config: TrainingConfig = None,
                            preprocessing_mode: str = "sparse",
                            # New time series parameters
                            align_to_daily: bool = False,
                            split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Prepare training and validation data loaders with proper time-based splitting
        
        Args:
            batch_size: Batch size for training
            validation_split: Fraction for validation (legacy, ignored in favor of split_ratios)
            sequence_length: Length of training sequences
            training_config: Training configuration object
            align_to_daily: If True, align sequences to trading day boundaries
            split_ratios: (train_ratio, val_ratio, test_ratio) for time-based splitting
            
        Returns:
            Tuple of (train_loader, val_loader) with proper time-based splits
        """
        if not self.current_portfolio:
            raise ValueError("No portfolio selected. Call select_portfolio() first.")
        
        self.logger.info("Preparing training data...")
        
        try:
            positions = self.current_portfolio['positions']
            
            # Log data mode selection
            alignment_str = "daily-aligned" if align_to_daily else "sliding window"
            self.logger.info(f"Using precomputed Greeks data ({preprocessing_mode} mode) with {alignment_str} sequences")
            self.logger.info(f"Time-based split ratios: train={split_ratios[0]:.1f}, val={split_ratios[1]:.1f}, test={split_ratios[2]:.1f}")
            
            # Create training data loader with time-based split
            train_loader = create_delta_data_loader(
                batch_size=batch_size,
                option_positions=positions,
                sequence_length=sequence_length,
                preprocessing_mode=preprocessing_mode,
                data_split='train',
                split_ratios=split_ratios,
                align_to_daily=align_to_daily,
                min_daily_sequences=getattr(training_config, 'min_daily_sequences', 1)
            )
            
            # Create validation data loader (smaller batch size for memory efficiency)
            # Create validation loader only if validation split > 0
            val_loader = None
            if len(split_ratios) > 1 and split_ratios[1] > 0:
                # Use validation_batch_size_factor from training config if available
                val_batch_size_factor = getattr(training_config, 'validation_batch_size_factor', 0.5)
                val_batch_size = max(1, int(batch_size * val_batch_size_factor))
                try:
                    val_loader = create_delta_data_loader(
                        batch_size=val_batch_size,
                        option_positions=positions,
                        sequence_length=sequence_length,
                        preprocessing_mode=preprocessing_mode,
                        data_split='val',
                        split_ratios=split_ratios,
                        align_to_daily=align_to_daily,
                        min_daily_sequences=getattr(training_config, 'min_daily_sequences', 1)
                    )
                except ValueError as e:
                    if "num_samples=0" in str(e):
                        self.logger.warning("No validation data available, training without validation")
                        val_loader = None
                    else:
                        raise
            
            self.logger.info(f"Training data loader created: batch_size={batch_size}")
            if val_loader is not None:
                val_batch_size = max(1, int(batch_size * getattr(training_config, 'validation_batch_size_factor', 0.5)))
                self.logger.info(f"Validation data loader created: batch_size={val_batch_size}")
            else:
                self.logger.info("No validation data loader created (validation split = 0)")
            
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def create_model(self, model_config: Optional[Dict] = None) -> PolicyNetwork:
        """
        Create and initialize policy network
        
        Args:
            model_config: Model configuration dict
            
        Returns:
            Initialized PolicyNetwork
        """
        if not self.current_portfolio:
            raise ValueError("No portfolio selected. Call select_portfolio() first.")
        
        # Default model configuration
        default_config = {
            'hidden_dims': [128, 64, 32],
            'dropout_rate': 0.1
        }
        
        if model_config:
            default_config.update(model_config)
        
        # Calculate input dimension using centralized dimension management
        from core.dimensions import get_dimension_calculator
        
        # For now, assume single asset (can be extended for multi-asset)  
        n_assets = 1
        
        # Get dimension calculator and compute input dimensions
        dim_calculator = get_dimension_calculator(n_assets)
        dim_spec = dim_calculator.calculate_model_input_dim(
            include_history=True,
            include_time=True, 
            include_delta=True
        )
        
        input_dim = dim_spec.total
        
        self.logger.info(f"Model input dimensions calculated: {dim_spec}")
        self.logger.debug(f"Dimension breakdown: {dim_spec.to_dict()}")
        
        self.logger.info(f"Creating model with input_dim={input_dim}")
        self.logger.info(f"Model config: {default_config}")
        
        model = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=default_config['hidden_dims'],
            dropout_rate=default_config['dropout_rate']
        )
        
        model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        return model
    
    def train(self, 
              config_name: str,
              training_config: TrainingConfig,
              enable_tracking: bool = True,
              enable_visualization: bool = True,
              model_config: Optional[Dict] = None,
              sequence_length: int = 100,
              validation_split: float = 0.2,
              preprocessing_mode: str = "sparse") -> Dict[str, Any]:
        """
        Execute complete training workflow
        
        Args:
            config_name: Portfolio configuration name
            training_config: Training configuration object (required)
            enable_tracking: Enable policy decision tracking
            enable_visualization: Enable real-time visualization
            model_config: Model architecture configuration
            sequence_length: Length of training sequences
            validation_split: Validation data split ratio
            
        Returns:
            Training results dictionary
        """
        
        training_start_time = time.time()
        
        try:
            # Step 1: Configuration Management
            self.logger.info("="*60)
            self.logger.info("STARTING PRODUCTION TRAINING WORKFLOW")
            self.logger.info("="*60)
            
            # Ensure we have a valid training configuration
            if training_config is None:
                raise ValueError("training_config is required. Use create_training_config() to generate a proper configuration.")
            
            # Select portfolio (this is always needed for data preparation)
            portfolio_info = self.select_portfolio(config_name)
            self.current_config = training_config
            
            # Step 2: Data Preparation
            train_loader, val_loader = self.prepare_training_data(
                batch_size=training_config.batch_size,
                sequence_length=sequence_length,
                validation_split=validation_split,
                training_config=training_config,
                preprocessing_mode=preprocessing_mode,
                align_to_daily=training_config.align_to_daily,
                split_ratios=training_config.split_ratios
            )
            
            # Step 3: Model Creation
            model = self.create_model(model_config)
            
            # Step 4: Trainer Setup
            self.current_trainer = Trainer(
                policy_network=model,
                training_config=training_config,
                device=self.device,
                checkpoint_dir=str(self.checkpoint_dir),
                enable_tracking=enable_tracking
            )
            
            # Step 5: Monitoring Setup
            hyperparameters = {
                'epochs': training_config.n_epochs,
                'batch_size': training_config.batch_size,
                'learning_rate': training_config.learning_rate,
                'sequence_length': getattr(training_config, 'sequence_length', 100),
                'checkpoint_interval': training_config.checkpoint_interval,
                'entropy_weight': training_config.entropy_weight,
                'base_execution_cost': training_config.base_execution_cost,
                'variable_execution_cost': training_config.variable_execution_cost
            }
            
            # Configure alert thresholds
            alert_thresholds = {
                'max_loss': 100.0,  # Loss explosion threshold
                'loss_stagnation_window': 10,
                'min_loss_std': 1e-6,
                'max_hedge_ratio': 0.9,
                'min_entropy': 0.1,  # Low exploration warning
                'max_execution_frequency': 0.95,  # High execution frequency warning
                'max_epoch_time': 300  # 5 minutes per epoch
            }
            
            self.training_monitor.alert_thresholds = alert_thresholds
            
            # Start monitoring session
            self.training_monitor.start_session(
                portfolio_config=config_name,
                model_config=model_config or {},
                hyperparameters=hyperparameters,
                total_epochs=training_config.n_epochs
            )
            
            # Step 6: Training Loop with Monitoring
            self.logger.info("Starting training loop...")
            
            history = []
            best_val_loss = float('inf')
            
            for epoch in range(training_config.n_epochs):
                epoch_start_time = time.time()
                
                # Training step
                train_metrics = self.current_trainer.train_epoch(train_loader)
                self.logger.info(f"[INFO] Epoch {epoch + 1} training completed - Loss: {train_metrics['loss']:.6f}")
                
                # Validation step (controlled by validation_interval and data availability)
                validation_interval = getattr(training_config, 'validation_interval', 1)
                if (epoch + 1) % validation_interval == 0 and val_loader is not None:
                    self.logger.info("[INFO] Starting validation...")
                    val_metrics = self.current_trainer.evaluate(val_loader)
                    self.logger.info(f"[INFO] Validation completed - Loss: {val_metrics['loss']:.6f}")
                else:
                    # Skip validation - either due to interval or no validation data
                    val_metrics = {'loss': float('inf')}
                    if val_loader is None:
                        self.logger.info("[INFO] Skipping validation (no validation data)")
                    else:
                        self.logger.info(f"[INFO] Skipping validation (interval={validation_interval})")
                        # Use previous validation metrics if available
                        if history and any(k.startswith('val_') for k in history[-1].keys()):
                            prev_val_metrics = {k.replace('val_', ''): v for k, v in history[-1].items() if k.startswith('val_')}
                            if prev_val_metrics:
                                val_metrics = prev_val_metrics
                
                epoch_time = time.time() - epoch_start_time
                
                # Combine metrics
                combined_metrics = {**train_metrics}
                for key, value in val_metrics.items():
                    combined_metrics[f'val_{key}'] = value
                
                history.append(combined_metrics)
                
                # Get policy behavior metrics if tracking enabled
                policy_metrics = {}
                if enable_tracking:
                    summary = global_policy_tracker.get_hedge_summary()
                    policy_metrics = {
                        'hedge_ratio': summary.get('hedge_ratio', 0),
                        'avg_execution_prob': summary.get('avg_execution_prob', 0),
                        'total_cost': summary.get('total_cost', 0)
                    }
                
                # Create training metrics object
                training_metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    step=epoch + 1,
                    timestamp=datetime.now(),
                    loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    learning_rate=training_config.learning_rate,
                    epoch_time=epoch_time,
                    **policy_metrics
                )
                
                # Log to monitor
                self.training_monitor.log_metrics(training_metrics)
                
                # Checkpoint saving
                if (epoch + 1) % training_config.checkpoint_interval == 0:
                    self.logger.info("[INFO] Saving checkpoint...")
                    self.current_trainer.save_checkpoint(epoch, history)
                    self.logger.info("[INFO] Checkpoint saved")
                
                # Best model tracking
                current_val_loss = val_metrics['loss']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    self.logger.info("[INFO] Saving new best model...")
                    self.current_trainer.save_checkpoint(epoch, history, 'best_model.pth')
                    self.logger.info(f"[INFO] New best model saved (val_loss: {current_val_loss:.6f})")
            
            # Step 7: Training Completion
            training_time = time.time() - training_start_time
            
            # Save final checkpoint
            self.logger.info("[INFO] Saving final checkpoint...")
            self.current_trainer.save_checkpoint(training_config.n_epochs - 1, history, 'final_model.pth')
            self.logger.info("[INFO] Final checkpoint saved")
            
            # End monitoring session
            self.training_monitor.end_session("completed")
            
            # Step 8: Generate Results and Visualizations
            results = self._generate_training_results(
                portfolio_info, hyperparameters, history, training_time, 
                enable_visualization
            )
            
            self.logger.info("="*60)
            self.logger.info("TRAINING WORKFLOW COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            self.logger.info(f"Total training time: {training_time:.1f} seconds")
            self.logger.info(f"Final training loss: {history[-1]['loss']:.6f}")
            self.logger.info(f"Final validation loss: {history[-1]['val_loss']:.6f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training workflow failed: {e}")
            self.logger.error(traceback.format_exc())
            
            # End monitoring session with error status
            if hasattr(self, 'training_monitor'):
                self.training_monitor.end_session("failed")
            
            raise
    
    def _generate_training_results(self, 
                                 portfolio_info: Dict,
                                 hyperparameters: Dict,
                                 history: list,
                                 training_time: float,
                                 enable_visualization: bool) -> Dict[str, Any]:
        """Generate comprehensive training results"""
        
        self.logger.info("Generating training results and visualizations...")
        
        results = {
            'portfolio_info': portfolio_info,
            'hyperparameters': hyperparameters,
            'training_history': history,
            'training_time_seconds': training_time,
            'final_metrics': history[-1] if history else {},
            'model_path': str(self.checkpoint_dir / 'final_model.pth'),
            'best_model_path': str(self.checkpoint_dir / 'best_model.pth')
        }
        
        # Add performance summary
        if history:
            losses = [h['loss'] for h in history]
            val_losses = [h['val_loss'] for h in history]
            
            results['performance_summary'] = {
                'final_loss': losses[-1],
                'best_loss': min(losses),
                'final_val_loss': val_losses[-1],
                'best_val_loss': min(val_losses),
                'training_epochs': len(history),
                'convergence_epoch': losses.index(min(losses)) + 1
            }
        
        # Generate visualizations if enabled
        if enable_visualization:
            try:
                viz_dir = self.output_dir / f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                viz_dir.mkdir(exist_ok=True)
                
                # Policy visualization
                visualizer = create_policy_visualizer()
                visualizer.export_visualizations(str(viz_dir))
                
                # Training monitor plots
                training_plots_path = viz_dir / 'training_monitor_plots.png'
                training_fig = self.training_monitor.generate_training_plots(str(training_plots_path))
                if training_fig:
                    plt.close(training_fig)  # Clean up matplotlib figure
                
                # Save training summary report
                summary_report = visualizer.generate_summary_report()
                with open(viz_dir / 'training_summary_report.txt', 'w') as f:
                    f.write(summary_report)
                
                results['visualization_dir'] = str(viz_dir)
                self.logger.info(f"Visualizations saved to: {viz_dir}")
                
            except Exception as e:
                self.logger.warning(f"Visualization generation failed: {e}")
        
        # Save results to file
        results_file = self.output_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        from dataclasses import asdict
        
        # Custom JSON encoder to handle datetime and other objects
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    try:
                        return asdict(obj)
                    except:
                        return str(obj)
                else:
                    return str(obj)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, cls=CustomJSONEncoder)
        
        results['results_file'] = str(results_file)
        self.logger.info(f"Training results saved to: {results_file}")
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status"""
        status = {
            'device': self.device,
            'has_active_trainer': self.current_trainer is not None,
            'current_portfolio': self.current_portfolio['config_name'] if self.current_portfolio else None
        }
        
        # Add monitoring status
        monitor_status = self.training_monitor.get_current_status()
        status.update(monitor_status)
        
        return status

# Global instance for easy access
production_trainer = ProductionTrainer()