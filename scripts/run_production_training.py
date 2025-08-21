#!/usr/bin/env python3
"""
Production Training Script for Gamma Hedging System

This script provides a complete production-ready interface for training
gamma hedging policies with real option data, comprehensive monitoring,
and professional-grade workflow management.

Usage:
    # Basic training
    python run_production_training.py --portfolio single_atm_call --epochs 50
    
    # Advanced training with time series features
    python run_production_training.py --portfolio single_atm_call --epochs 50 \\
        --align-to-daily --split-ratios 0.6 0.3 0.1 --underlying-dense-mode
    
    # Custom configuration
    python run_production_training.py --portfolio best_liquidity_single \\
        --batch-size 64 --lr 0.002 --min-daily-sequences 100
    
    # List available portfolios
    python run_production_training.py --list-portfolios
"""

import argparse
import sys
import os
import logging
import json
from pathlib import Path
from typing import Dict, Optional

# Add project root to path (script is now in scripts/ subdirectory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import production training components
from training.production_trainer import ProductionTrainer
from training.config import TrainingConfig
from data.option_portfolio_manager import OptionPortfolioManager
from training.training_monitor import TrainingMonitor

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Production Gamma Hedging Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --portfolio single_atm_call --epochs 50
  %(prog)s --portfolio best_liquidity_single --batch-size 64 --lr 0.002
  %(prog)s --list-portfolios
  %(prog)s --portfolio-info single_atm_call
        """
    )
    
    # Portfolio selection
    portfolio_group = parser.add_argument_group('Portfolio Selection')
    portfolio_group.add_argument('--portfolio', '-p', type=str,
                                help='Portfolio configuration name to use for training')
    portfolio_group.add_argument('--list-portfolios', action='store_true',
                                help='List available portfolio configurations and exit')
    portfolio_group.add_argument('--portfolio-info', type=str,
                                help='Show detailed info for a portfolio configuration and exit')
    portfolio_group.add_argument('--portfolio-config', type=str,
                                help='Path to custom portfolio configuration file')
    
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--epochs', '-e', type=int, default=50,
                               help='Number of training epochs (default: 50)')
    training_group.add_argument('--batch-size', '-b', type=int, default=32,
                               help='Batch size for training (default: 32)')
    training_group.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                               help='Learning rate (default: 0.001)')
    training_group.add_argument('--sequence-length', type=int, default=100,
                               help='Sequence length for time series (default: 100)')
    training_group.add_argument('--checkpoint-interval', type=int, default=10,
                               help='Save checkpoint every N epochs (default: 10)')
    training_group.add_argument('--validation-split', type=float, default=0.2,
                               help='Validation data split ratio (default: 0.2)')
    training_group.add_argument('--underlying-dense-mode', action='store_true',
                               help='Use underlying data density for training (greatly increases data volume)')
    training_group.add_argument('--auto-dense-mode', action='store_true', default=True,
                               help='Automatically enable dense mode if sparse data is insufficient (default: True)')
    
    # Time series parameters
    timeseries_group = parser.add_argument_group('Time Series Configuration')
    timeseries_group.add_argument('--align-to-daily', action='store_true',
                                 help='Align sequences to trading day boundaries (enables variable-length sequences)')
    timeseries_group.add_argument('--split-ratios', nargs=3, type=float, default=[0.7, 0.2, 0.1],
                                 metavar=('TRAIN', 'VAL', 'TEST'),
                                 help='Time-based split ratios for train/val/test (default: 0.7 0.2 0.1)')
    timeseries_group.add_argument('--min-daily-sequences', type=int, default=50,
                                 help='Minimum data points required per trading day (default: 50)')
    timeseries_group.add_argument('--disable-time-split', action='store_true',
                                 help='Disable time-based splitting (legacy mode)')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--hidden-dims', nargs='+', type=int, default=[128, 64, 32],
                            help='Hidden layer dimensions (default: 128 64 32)')
    model_group.add_argument('--dropout-rate', type=float, default=0.1,
                            help='Dropout rate (default: 0.1)')
    
    # Monitoring and visualization
    monitor_group = parser.add_argument_group('Monitoring and Visualization')
    monitor_group.add_argument('--enable-tracking', action='store_true', default=True,
                              help='Enable policy decision tracking (default: True)')
    monitor_group.add_argument('--disable-tracking', action='store_true',
                              help='Disable policy decision tracking')
    monitor_group.add_argument('--enable-visualization', action='store_true', default=True,
                              help='Enable real-time visualization (default: True)')
    monitor_group.add_argument('--disable-visualization', action='store_true',
                              help='Disable real-time visualization')
    monitor_group.add_argument('--disable-realtime-plots', action='store_true',
                              help='Disable real-time plots (keep other monitoring)')
    
    # Output and logging
    output_group = parser.add_argument_group('Output and Logging')
    output_group.add_argument('--output-dir', '-o', type=str, default='outputs',
                             help='Output directory for results (default: outputs)')
    output_group.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                             help='Checkpoint directory (default: checkpoints)')
    output_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                             default='INFO', help='Logging level (default: INFO)')
    output_group.add_argument('--log-file', type=str,
                             help='Log file path (default: console only)')
    
    # System configuration
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                             help='Device to use for training (default: auto)')
    system_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def resolve_device(device_arg: str) -> str:
    """Resolve device specification"""
    import torch
    
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_arg == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        return 'cpu'
    else:
        return device_arg

def list_portfolios(portfolio_manager: OptionPortfolioManager):
    """List available portfolio configurations"""
    
    print("\nAvailable Portfolio Configurations:")
    print("=" * 50)
    
    configs = portfolio_manager.list_available_configs()
    
    for config_name in configs:
        try:
            info = portfolio_manager.get_config_info(config_name)
            config = info['config']
            
            print(f"\n{config_name}:")
            print(f"  Name: {config['name']}")
            print(f"  Description: {config['description']}")
            print(f"  Max Options: {config['max_options']}")
            print(f"  Matching Options: {info['matching_options']}")
            
            if info['sample_options']:
                print("  Sample Options:")
                for option in info['sample_options'][:3]:
                    print(f"    - {option}")
                if len(info['sample_options']) > 3:
                    print(f"    ... and {len(info['sample_options']) - 3} more")
            
        except Exception as e:
            print(f"  Error loading config: {e}")
    
    print(f"\nTotal: {len(configs)} portfolio configurations available")

def show_portfolio_info(portfolio_manager: OptionPortfolioManager, config_name: str):
    """Show detailed information for a specific portfolio"""
    
    try:
        report = portfolio_manager.generate_portfolio_report(config_name)
        print(report)
        
    except Exception as e:
        print(f"Error generating portfolio report: {e}")
        sys.exit(1)

def create_training_config(portfolio_manager: OptionPortfolioManager, 
                          portfolio_name: str, 
                          cli_args) -> TrainingConfig:
    """
    Create unified training configuration with clear priority hierarchy:
    CLI Arguments > Portfolio Template > TrainingConfig Defaults
    
    This function eliminates parameter conflicts and ensures consistent configuration.
    """
    
    # Step 1: Start with TrainingConfig defaults
    training_config = TrainingConfig()
    
    # Step 2: Load and apply portfolio template parameters
    portfolio_config = portfolio_manager.load_portfolio_config(portfolio_name)
    
    # Apply training_params from portfolio template
    training_params = portfolio_config.get('training_params', {})
    for key, value in training_params.items():
        if hasattr(training_config, key):
            setattr(training_config, key, value)
    
    # Apply time_series_params from portfolio template  
    time_series_params = portfolio_config.get('time_series_params', {})
    for key, value in time_series_params.items():
        if hasattr(training_config, key):
            setattr(training_config, key, value)
    
    # Step 3: Apply CLI arguments (highest priority)
    # Define parameter mapping for CLI args that have different names
    cli_overrides = {
        # Core training parameters
        'epochs': 'n_epochs',
        'batch_size': 'batch_size', 
        'learning_rate': 'learning_rate',
        'checkpoint_interval': 'checkpoint_interval',
        
        # Time series parameters
        'split_ratios': 'split_ratios',
        'min_daily_sequences': 'min_daily_sequences'
    }
    
    # Apply CLI overrides only if the argument was actually provided
    for cli_param, config_param in cli_overrides.items():
        if hasattr(cli_args, cli_param):
            cli_value = getattr(cli_args, cli_param)
            if cli_value is not None:  # Only override if value was provided
                if cli_param == 'split_ratios':
                    setattr(training_config, config_param, tuple(cli_value))
                else:
                    setattr(training_config, config_param, cli_value)
    
    return training_config

def main():
    """Main training workflow"""
    
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Resolve device
    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize portfolio manager
        portfolio_manager = OptionPortfolioManager()
        
        # Handle information commands
        if args.list_portfolios:
            list_portfolios(portfolio_manager)
            return
        
        if args.portfolio_info:
            show_portfolio_info(portfolio_manager, args.portfolio_info)
            return
        
        # Validate portfolio selection
        if not args.portfolio:
            logger.error("Portfolio configuration required. Use --portfolio or --list-portfolios")
            sys.exit(1)
        
        # Load custom portfolio config if provided
        if args.portfolio_config:
            config_name = portfolio_manager.load_portfolio_config(args.portfolio_config)
            logger.info(f"Loaded custom portfolio config: {config_name}")
            args.portfolio = config_name
        
        # Validate portfolio exists
        available_configs = portfolio_manager.list_available_configs()
        if args.portfolio not in available_configs:
            logger.error(f"Unknown portfolio: {args.portfolio}")
            logger.error(f"Available portfolios: {', '.join(available_configs)}")
            sys.exit(1)
        
        # Handle tracking and visualization flags
        enable_tracking = args.enable_tracking and not args.disable_tracking
        enable_visualization = args.enable_visualization and not args.disable_visualization
        enable_realtime_plots = not args.disable_realtime_plots
        
        # Initialize training monitor
        training_monitor = TrainingMonitor(
            log_dir=os.path.join(args.output_dir, "logs"),
            enable_realtime_plots=enable_realtime_plots and enable_visualization
        )
        
        # Initialize production trainer
        production_trainer = ProductionTrainer(
            portfolio_manager=portfolio_manager,
            training_monitor=training_monitor,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir
        )
        
        # Prepare model configuration
        model_config = {
            'hidden_dims': args.hidden_dims,
            'dropout_rate': args.dropout_rate
        }
        
        # Display training configuration
        logger.info("=" * 60)
        logger.info("PRODUCTION TRAINING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Portfolio: {args.portfolio}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info(f"Sequence Length: {args.sequence_length}")
        logger.info(f"Model Hidden Dims: {args.hidden_dims}")
        logger.info(f"Dropout Rate: {args.dropout_rate}")
        logger.info(f"Device: {device}")
        logger.info(f"Enable Tracking: {enable_tracking}")
        logger.info(f"Enable Visualization: {enable_visualization}")
        logger.info(f"Checkpoint Interval: {args.checkpoint_interval}")
        logger.info(f"Random Seed: {args.seed}")
        logger.info(f"Underlying Dense Mode: {args.underlying_dense_mode}")
        logger.info(f"Auto Dense Mode: {args.auto_dense_mode}")
        
        # Determine data mode  
        underlying_dense_mode = args.underlying_dense_mode
        if args.auto_dense_mode and not underlying_dense_mode:
            # Auto-enable dense mode for better training stability
            underlying_dense_mode = True
            logger.info("Auto-enabling underlying dense mode for optimal training data")
        
        # Validate split ratios
        if abs(sum(args.split_ratios) - 1.0) > 1e-6:
            logger.error(f"Split ratios must sum to 1.0, got: {args.split_ratios} (sum={sum(args.split_ratios)})")
            return 1
        
        # Create unified training configuration with clear priority: CLI > Portfolio > Defaults
        training_config = create_training_config(
            portfolio_manager=production_trainer.portfolio_manager,
            portfolio_name=args.portfolio,
            cli_args=args
        )
        
        logger.info(f"Final Training Configuration:")
        logger.info(f"  Epochs: {training_config.n_epochs}")
        logger.info(f"  Batch Size: {training_config.batch_size}")
        logger.info(f"  Learning Rate: {training_config.learning_rate}")
        logger.info(f"  Align to Daily: {training_config.align_to_daily}")
        logger.info(f"  Split Ratios: {training_config.split_ratios}")
        logger.info(f"  Min Daily Sequences: {training_config.min_daily_sequences}")
        
        # Execute training with complete training_config
        results = production_trainer.train(
            config_name=args.portfolio,
            training_config=training_config,
            enable_tracking=enable_tracking,
            enable_visualization=enable_visualization,
            model_config=model_config,
            sequence_length=args.sequence_length,
            validation_split=args.validation_split,
            underlying_dense_mode=underlying_dense_mode
        )
        
        # Display results summary
        logger.info("=" * 60)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("=" * 60)
        
        if 'performance_summary' in results:
            perf = results['performance_summary']
            logger.info(f"Training Epochs: {perf['training_epochs']}")
            logger.info(f"Final Training Loss: {perf['final_loss']:.6f}")
            logger.info(f"Final Validation Loss: {perf['final_val_loss']:.6f}")
            logger.info(f"Best Training Loss: {perf['best_loss']:.6f}")
            logger.info(f"Best Validation Loss: {perf['best_val_loss']:.6f}")
            logger.info(f"Convergence Epoch: {perf['convergence_epoch']}")
        
        logger.info(f"Training Time: {results['training_time_seconds']:.1f} seconds")
        logger.info(f"Results File: {results['results_file']}")
        
        if 'visualization_dir' in results:
            logger.info(f"Visualizations: {results['visualization_dir']}")
        
        logger.info(f"Model Checkpoints: {args.checkpoint_dir}")
        logger.info(f"Best Model: {results['best_model_path']}")
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()