import time
import logging
import traceback
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import torch
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import existing components
from models.policy_network import PolicyNetwork
from training.trainer import Trainer
from training.training_monitor import TrainingMonitor, TrainingMetrics
from training.config import TrainingConfig
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
        self.logger = logging.getLogger(__name__)
        
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
                            sequence_length: int = 100) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Prepare training and validation data loaders
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if not self.current_portfolio:
            raise ValueError("No portfolio selected. Call select_portfolio() first.")
        
        self.logger.info("Preparing training data...")
        
        try:
            positions = self.current_portfolio['positions']
            
            # Create training data loader
            train_loader = create_delta_data_loader(
                batch_size=batch_size,
                option_positions=positions
            )
            
            # Create validation data loader (smaller batch size for memory efficiency)
            val_batch_size = max(1, batch_size // 2)
            val_loader = create_delta_data_loader(
                batch_size=val_batch_size,
                option_positions=positions
            )
            
            self.logger.info(f"Training data loader created: batch_size={batch_size}")
            self.logger.info(f"Validation data loader created: batch_size={val_batch_size}")
            
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
        
        # Calculate input dimension
        # For now, assume single asset (can be extended for multi-asset)
        n_assets = 1
        input_dim = n_assets * 3 + 1  # prev_holding, current_holding, price + time_features
        
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
              training_config: Optional[TrainingConfig] = None,
              enable_tracking: bool = True,
              enable_visualization: bool = True,
              model_config: Optional[Dict] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Execute complete training workflow
        
        Args:
            config_name: Portfolio configuration name
            training_config: Training configuration (will be created from portfolio template if None)
            enable_tracking: Enable policy decision tracking
            enable_visualization: Enable real-time visualization
            model_config: Model architecture configuration
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        
        training_start_time = time.time()
        
        try:
            # Step 1: Configuration Management
            self.logger.info("="*60)
            self.logger.info("STARTING PRODUCTION TRAINING WORKFLOW")
            self.logger.info("="*60)
            
            # Create or merge training configuration
            if training_config is None:
                # Load configuration from portfolio template
                portfolio_info = self.select_portfolio(config_name)
                portfolio_config = self.portfolio_manager.load_portfolio_config(config_name)
                training_params = portfolio_config.get('training_params', {})
                
                # Create TrainingConfig from portfolio template
                training_config = TrainingConfig()
                for key, value in training_params.items():
                    if hasattr(training_config, key):
                        setattr(training_config, key, value)
                
                # Override with any kwargs (handle parameter name mapping)
                param_mapping = {
                    'epochs': 'n_epochs',  # Map epochs -> n_epochs
                }
                
                for key, value in kwargs.items():
                    # Use mapped key if available, otherwise use original key
                    config_key = param_mapping.get(key, key)
                    if hasattr(training_config, config_key):
                        setattr(training_config, config_key, value)
            else:
                # Use provided configuration but still need portfolio info
                portfolio_info = self.select_portfolio(config_name)
            
            self.current_config = training_config
            
            # Step 2: Data Preparation
            train_loader, val_loader = self.prepare_training_data(
                batch_size=training_config.batch_size,
                sequence_length=getattr(training_config, 'sequence_length', 100),
                validation_split=kwargs.get('validation_split', 0.2)
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
                
                # Validation step
                val_metrics = self.current_trainer.evaluate(val_loader)
                
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
                    self.current_trainer.save_checkpoint(epoch, history)
                
                # Best model tracking
                current_val_loss = val_metrics['loss']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    self.current_trainer.save_checkpoint(epoch, history, 'best_model.pth')
                    self.logger.info(f"New best model saved (val_loss: {current_val_loss:.6f})")
            
            # Step 7: Training Completion
            training_time = time.time() - training_start_time
            
            # Save final checkpoint
            self.current_trainer.save_checkpoint(training_config.n_epochs - 1, history, 'final_model.pth')
            
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