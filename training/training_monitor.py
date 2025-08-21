import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import threading
from collections import deque

@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific point in time"""
    epoch: int
    step: int
    timestamp: datetime
    loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.001
    
    # Policy-specific metrics
    avg_execution_prob: Optional[float] = None
    hedge_ratio: Optional[float] = None
    total_cost: Optional[float] = None
    
    # Regularization metrics
    entropy: Optional[float] = None
    policy_loss: Optional[float] = None
    entropy_loss: Optional[float] = None
    execution_frequency: Optional[float] = None
    
    # Performance metrics
    batch_time: Optional[float] = None
    epoch_time: Optional[float] = None
    gpu_memory: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class TrainingSession:
    """Information about the current training session"""
    session_id: str
    start_time: datetime
    portfolio_config: str
    model_config: Dict
    hyperparameters: Dict
    total_epochs: int
    status: str = "running"  # running, completed, failed, stopped
    end_time: Optional[datetime] = None

class TrainingMonitor:
    """
    Comprehensive training monitoring system with real-time metrics,
    logging, visualization, and alerting capabilities.
    """
    
    def __init__(self, 
                 log_dir: str = "outputs/logs",
                 enable_realtime_plots: bool = False,  # Disabled by default to avoid threading issues
                 metrics_window_size: int = 100,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_realtime_plots = enable_realtime_plots
        self.metrics_window_size = metrics_window_size
        self.alert_thresholds = alert_thresholds or {}
        
        # Training state
        self.current_session: Optional[TrainingSession] = None
        self.metrics_history: List[TrainingMetrics] = []
        self.recent_metrics = deque(maxlen=metrics_window_size)
        
        # Real-time monitoring
        self.last_update_time = time.time()
        self.monitoring_active = False
        self.plot_update_thread = None
        self.alert_callbacks: List[Callable] = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize plots if enabled
        if self.enable_realtime_plots:
            self._setup_realtime_plots()
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        self.logger = logging.getLogger("TrainingMonitor")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for training logs
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Training monitor initialized. Logs: {log_file}")
    
    def _setup_realtime_plots(self):
        """Setup real-time plotting infrastructure"""
        # Note: Realtime plots disabled to prevent threading issues
        # Use generate_training_plots() instead for post-training visualization
        self.logger.warning("Real-time plots disabled to prevent threading issues. Use generate_training_plots() after training.")
        return
        
        # Configure subplots
        self.axes[0, 0].set_title('Training Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title('Policy Behavior')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Hedge Ratio')
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title('Training Speed')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Time per Epoch (s)')
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title('Cost Performance')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Total Cost')
        self.axes[1, 1].grid(True)
        
        # Initialize empty lines
        self.loss_line, = self.axes[0, 0].plot([], [], 'b-', label='Training Loss')
        self.val_loss_line, = self.axes[0, 0].plot([], [], 'r-', label='Validation Loss')
        self.axes[0, 0].legend()
        
        self.hedge_ratio_line, = self.axes[0, 1].plot([], [], 'g-', label='Hedge Ratio')
        self.exec_prob_line, = self.axes[0, 1].plot([], [], 'orange', alpha=0.7, label='Avg Execution Prob')
        self.axes[0, 1].legend()
        
        self.epoch_time_line, = self.axes[1, 0].plot([], [], 'm-', label='Epoch Time')
        self.axes[1, 0].legend()
        
        self.cost_line, = self.axes[1, 1].plot([], [], 'c-', label='Total Cost')
        self.axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show(block=False)
    
    def start_session(self, 
                     portfolio_config: str,
                     model_config: Dict,
                     hyperparameters: Dict,
                     total_epochs: int):
        """Start a new training session"""
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now(),
            portfolio_config=portfolio_config,
            model_config=model_config,
            hyperparameters=hyperparameters,
            total_epochs=total_epochs
        )
        
        # Reset metrics
        self.metrics_history = []
        self.recent_metrics.clear()
        
        self.monitoring_active = True
        
        # Save session info
        self._save_session_info()
        
        self.logger.info(f"Started training session: {session_id}")
        self.logger.info(f"Portfolio: {portfolio_config}")
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Hyperparameters: {hyperparameters}")
        
        # Real-time plot updates disabled to prevent threading issues
        # Use generate_training_plots() instead
        if self.enable_realtime_plots:
            self.logger.warning("Real-time plots requested but disabled for thread safety. Use generate_training_plots() after training.")
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        
        if not self.current_session:
            self.logger.warning("No active training session")
            return
        
        # Add to history
        self.metrics_history.append(metrics)
        self.recent_metrics.append(metrics)
        
        # Update last update time
        self.last_update_time = time.time()
        
        # Log key metrics
        log_msg = f"Epoch {metrics.epoch:3d} | Loss: {metrics.loss:.6f}"
        
        if metrics.val_loss is not None:
            log_msg += f" | Val Loss: {metrics.val_loss:.6f}"
        
        if metrics.hedge_ratio is not None:
            log_msg += f" | Hedge Ratio: {metrics.hedge_ratio:.3f}"
        
        if metrics.total_cost is not None:
            log_msg += f" | Total Cost: {metrics.total_cost:.2f}"
        
        if metrics.entropy is not None:
            log_msg += f" | Entropy: {metrics.entropy:.4f}"
            
        if metrics.execution_frequency is not None:
            log_msg += f" | Exec Freq: {metrics.execution_frequency:.3f}"
        
        if metrics.epoch_time is not None:
            log_msg += f" | Time: {metrics.epoch_time:.1f}s"
        
        self.logger.info(log_msg)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Save metrics to file
        self._save_metrics_checkpoint()
    
    def _check_alerts(self, metrics: TrainingMetrics):
        """Check for alert conditions and trigger callbacks"""
        
        alerts = []
        
        # Loss explosion
        if 'max_loss' in self.alert_thresholds:
            if metrics.loss > self.alert_thresholds['max_loss']:
                alerts.append(f"Loss explosion detected: {metrics.loss:.6f}")
        
        # Loss stagnation
        if 'loss_stagnation_window' in self.alert_thresholds:
            window_size = self.alert_thresholds['loss_stagnation_window']
            if len(self.recent_metrics) >= window_size:
                recent_losses = [m.loss for m in list(self.recent_metrics)[-window_size:]]
                loss_std = np.std(recent_losses)
                if loss_std < self.alert_thresholds.get('min_loss_std', 1e-6):
                    alerts.append(f"Loss stagnation detected (std: {loss_std:.8f})")
        
        # Extreme hedge ratio
        if metrics.hedge_ratio is not None:
            if 'max_hedge_ratio' in self.alert_thresholds:
                if metrics.hedge_ratio > self.alert_thresholds['max_hedge_ratio']:
                    alerts.append(f"Extreme hedge ratio: {metrics.hedge_ratio:.3f}")
        
        # Low entropy (lack of exploration)
        if metrics.entropy is not None:
            if 'min_entropy' in self.alert_thresholds:
                if metrics.entropy < self.alert_thresholds['min_entropy']:
                    alerts.append(f"Low entropy detected: {metrics.entropy:.4f}")
        
        # High execution frequency
        if metrics.execution_frequency is not None:
            if 'max_execution_frequency' in self.alert_thresholds:
                if metrics.execution_frequency > self.alert_thresholds['max_execution_frequency']:
                    alerts.append(f"High execution frequency: {metrics.execution_frequency:.3f}")
        
        # Training time alerts
        if metrics.epoch_time is not None:
            if 'max_epoch_time' in self.alert_thresholds:
                if metrics.epoch_time > self.alert_thresholds['max_epoch_time']:
                    alerts.append(f"Slow training detected: {metrics.epoch_time:.1f}s per epoch")
        
        # Trigger alert callbacks
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert}")
            for callback in self.alert_callbacks:
                try:
                    callback(alert, metrics)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
    
    def _start_plot_updates(self):
        """Start background thread for plot updates"""
        # Disabled to prevent threading issues with matplotlib
        self.logger.info("Real-time plot updates disabled. Use generate_training_plots() after training.")
        return
    
    def generate_training_plots(self, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Generate training plots after training completion (thread-safe)"""
        if not self.metrics_history:
            self.logger.warning("No metrics history available for plotting")
            return None
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Monitor - Metrics Summary', fontsize=16)
        
        # Prepare data
        epochs = [m.epoch for m in self.metrics_history]
        losses = [m.loss for m in self.metrics_history]
        
        # 1. Loss plot
        axes[0, 0].plot(epochs, losses, 'b-', label='Training Loss', linewidth=2)
        
        val_losses = [m.val_loss for m in self.metrics_history if m.val_loss is not None]
        if val_losses:
            val_epochs = [m.epoch for m in self.metrics_history if m.val_loss is not None]
            axes[0, 0].plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Policy behavior
        hedge_ratios = [m.hedge_ratio for m in self.metrics_history if m.hedge_ratio is not None]
        if hedge_ratios:
            hedge_epochs = [m.epoch for m in self.metrics_history if m.hedge_ratio is not None]
            axes[0, 1].plot(hedge_epochs, hedge_ratios, 'g-', label='Hedge Ratio', linewidth=2)
        
        exec_probs = [m.avg_execution_prob for m in self.metrics_history if m.avg_execution_prob is not None]
        if exec_probs:
            exec_epochs = [m.epoch for m in self.metrics_history if m.avg_execution_prob is not None]
            axes[0, 1].plot(exec_epochs, exec_probs, 'orange', alpha=0.7, label='Avg Execution Prob', linewidth=2)
        
        # Add entropy plot
        entropies = [m.entropy for m in self.metrics_history if m.entropy is not None]
        if entropies:
            entropy_epochs = [m.epoch for m in self.metrics_history if m.entropy is not None]
            axes[0, 1].plot(entropy_epochs, entropies, 'purple', alpha=0.8, label='Entropy', linewidth=2)
        
        axes[0, 1].set_title('Policy Behavior & Exploration')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Ratio / Probability / Entropy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training speed
        epoch_times = [m.epoch_time for m in self.metrics_history if m.epoch_time is not None]
        if epoch_times:
            time_epochs = [m.epoch for m in self.metrics_history if m.epoch_time is not None]
            axes[1, 0].plot(time_epochs, epoch_times, 'm-', label='Epoch Time', linewidth=2)
        
        axes[1, 0].set_title('Training Speed')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time per Epoch (s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cost performance
        costs = [m.total_cost for m in self.metrics_history if m.total_cost is not None]
        if costs:
            cost_epochs = [m.epoch for m in self.metrics_history if m.total_cost is not None]
            axes[1, 1].plot(cost_epochs, costs, 'c-', label='Total Cost', linewidth=2)
        
        axes[1, 1].set_title('Cost Performance')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Total Cost')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plots saved to {save_path}")
        
        return fig
    
    def end_session(self, status: str = "completed"):
        """End the current training session"""
        
        if not self.current_session:
            self.logger.warning("No active session to end")
            return
        
        self.current_session.status = status
        self.current_session.end_time = datetime.now()
        self.monitoring_active = False
        
        # Calculate session summary
        duration = self.current_session.end_time - self.current_session.start_time
        total_epochs_completed = len(self.metrics_history)
        
        self.logger.info(f"Training session ended: {status}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Epochs completed: {total_epochs_completed}/{self.current_session.total_epochs}")
        
        if self.metrics_history:
            final_loss = self.metrics_history[-1].loss
            self.logger.info(f"Final loss: {final_loss:.6f}")
        
        # Save final session data
        self._save_session_summary()
        
        # Keep plots open for inspection
        if self.enable_realtime_plots:
            self.logger.info("Real-time plots remain open for inspection")
    
    def _save_session_info(self):
        """Save session information to file"""
        if not self.current_session:
            return
        
        session_file = self.log_dir / f"{self.current_session.session_id}_info.json"
        
        session_data = {
            'session_id': self.current_session.session_id,
            'start_time': self.current_session.start_time.isoformat(),
            'portfolio_config': self.current_session.portfolio_config,
            'model_config': self.current_session.model_config,
            'hyperparameters': self.current_session.hyperparameters,
            'total_epochs': self.current_session.total_epochs,
            'status': self.current_session.status
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def _save_metrics_checkpoint(self):
        """Save metrics checkpoint"""
        if not self.current_session or not self.metrics_history:
            return
        
        metrics_file = self.log_dir / f"{self.current_session.session_id}_metrics.json"
        
        metrics_data = []
        for m in self.metrics_history:
            metric_dict = {
                'epoch': m.epoch,
                'step': m.step,
                'timestamp': m.timestamp.isoformat(),
                'loss': m.loss,
                'val_loss': m.val_loss,
                'learning_rate': m.learning_rate,
                'avg_execution_prob': m.avg_execution_prob,
                'hedge_ratio': m.hedge_ratio,
                'total_cost': m.total_cost,
                'entropy': m.entropy,
                'policy_loss': m.policy_loss,
                'entropy_loss': m.entropy_loss,
                'execution_frequency': m.execution_frequency,
                'batch_time': m.batch_time,
                'epoch_time': m.epoch_time,
                'gpu_memory': m.gpu_memory,
                'custom_metrics': m.custom_metrics
            }
            metrics_data.append(metric_dict)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def _save_session_summary(self):
        """Save comprehensive session summary"""
        if not self.current_session:
            return
        
        summary_file = self.log_dir / f"{self.current_session.session_id}_summary.json"
        
        # Calculate statistics
        if self.metrics_history:
            losses = [m.loss for m in self.metrics_history]
            summary_stats = {
                'total_epochs': len(self.metrics_history),
                'final_loss': losses[-1],
                'min_loss': min(losses),
                'max_loss': max(losses),
                'avg_loss': np.mean(losses),
                'loss_std': np.std(losses),
            }
            
            # Add policy-specific stats
            hedge_ratios = [m.hedge_ratio for m in self.metrics_history if m.hedge_ratio is not None]
            if hedge_ratios:
                summary_stats.update({
                    'avg_hedge_ratio': np.mean(hedge_ratios),
                    'final_hedge_ratio': hedge_ratios[-1]
                })
            
            costs = [m.total_cost for m in self.metrics_history if m.total_cost is not None]
            if costs:
                summary_stats.update({
                    'final_cost': costs[-1],
                    'best_cost': min(costs)
                })
            
            # Add regularization stats
            entropies = [m.entropy for m in self.metrics_history if m.entropy is not None]
            if entropies:
                summary_stats.update({
                    'avg_entropy': np.mean(entropies),
                    'final_entropy': entropies[-1],
                    'min_entropy': min(entropies),
                    'max_entropy': max(entropies)
                })
            
            exec_freqs = [m.execution_frequency for m in self.metrics_history if m.execution_frequency is not None]
            if exec_freqs:
                summary_stats.update({
                    'avg_execution_frequency': np.mean(exec_freqs),
                    'final_execution_frequency': exec_freqs[-1]
                })
        else:
            summary_stats = {}
        
        summary_data = {
            'session_info': {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time.isoformat(),
                'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                'duration_seconds': (self.current_session.end_time - self.current_session.start_time).total_seconds() if self.current_session.end_time else None,
                'status': self.current_session.status
            },
            'configuration': {
                'portfolio_config': self.current_session.portfolio_config,
                'hyperparameters': self.current_session.hyperparameters
            },
            'performance_summary': summary_stats
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        self.logger.info(f"Session summary saved: {summary_file}")
    
    def add_alert_callback(self, callback: Callable[[str, TrainingMetrics], None]):
        """Add a callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current training status"""
        if not self.current_session:
            return {'status': 'no_active_session'}
        
        status = {
            'session_id': self.current_session.session_id,
            'status': self.current_session.status,
            'epochs_completed': len(self.metrics_history),
            'total_epochs': self.current_session.total_epochs,
            'progress_percent': (len(self.metrics_history) / self.current_session.total_epochs) * 100,
            'time_elapsed': (datetime.now() - self.current_session.start_time).total_seconds(),
        }
        
        if self.metrics_history:
            latest = self.metrics_history[-1]
            status.update({
                'current_epoch': latest.epoch,
                'current_loss': latest.loss,
                'current_hedge_ratio': latest.hedge_ratio,
                'current_cost': latest.total_cost
            })
        
        return status

# Global instance for easy access
training_monitor = TrainingMonitor()