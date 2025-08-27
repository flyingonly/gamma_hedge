#!/usr/bin/env python3
"""
Unified Training Entry Point for Gamma Hedge System

This script provides a single, simplified entry point for all training scenarios,
replacing the complex logic in main.py and run_production_training.py with a 
clean, orchestrator-based approach.

Usage Examples:
    # Basic training
    python scripts/train.py --epochs 50 --batch-size 32
    
    # Delta hedging with portfolio
    python scripts/train.py --portfolio single_atm_call --epochs 50 --delta-hedge
    
    # Resume training
    python scripts/train.py --resume --epochs 100
    
    # Evaluation only
    python scripts/train.py --eval-only --model checkpoints/best_model.pth
    
    # Validation
    python scripts/train.py --validate --validation-type system
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import create_config
from core.orchestrator import ApplicationOrchestrator, WorkflowType
from data import PortfolioManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified Training System for Gamma Hedge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --epochs 50 --batch-size 32
  %(prog)s --portfolio single_atm_call --delta-hedge --epochs 50  
  %(prog)s --resume --epochs 100
  %(prog)s --eval-only --model checkpoints/best_model.pth
  %(prog)s --validate --validation-type all
        """
    )
    
    # Main operation mode
    operation_group = parser.add_mutually_exclusive_group()
    operation_group.add_argument('--train', action='store_true', default=True,
                                help='Training mode (default)')
    operation_group.add_argument('--eval-only', action='store_true',
                                help='Evaluation only mode')
    operation_group.add_argument('--validate', action='store_true',
                                help='Validation mode')
    operation_group.add_argument('--preprocess', action='store_true',
                                help='Preprocessing mode')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs', type=int, default=50,
                           help='Number of training epochs (default: 50)')
    train_group.add_argument('--batch-size', type=int, default=32,
                           help='Training batch size (default: 32)')
    train_group.add_argument('--learning-rate', type=float, default=0.001,
                           help='Learning rate (default: 0.001)')
    train_group.add_argument('--resume', action='store_true',
                           help='Resume training from latest checkpoint')
    train_group.add_argument('--checkpoint-interval', type=int, default=10,
                           help='Save checkpoint every N epochs (default: 10)')
    
    # Data and portfolio options
    data_group = parser.add_argument_group('Data and Portfolio Options')
    data_group.add_argument('--delta-hedge', action='store_true',
                          help='Enable delta hedging mode')
    data_group.add_argument('--portfolio', type=str,
                          help='Portfolio template name for delta hedging')
    data_group.add_argument('--list-portfolios', action='store_true',
                          help='List available portfolio templates and exit')
    data_group.add_argument('--portfolio-info', type=str,
                          help='Show information about a portfolio template')
    
    # Model and evaluation options
    model_group = parser.add_argument_group('Model and Evaluation Options') 
    model_group.add_argument('--model', type=str,
                           help='Model checkpoint path for evaluation')
    model_group.add_argument('--eval-samples', type=int, default=1000,
                           help='Number of samples for evaluation (default: 1000)')
    
    # Validation options
    validation_group = parser.add_argument_group('Validation Options')
    validation_group.add_argument('--validation-type', type=str, 
                                choices=['system', 'data', 'all'], default='system',
                                help='Type of validation to perform (default: system)')
    
    # Preprocessing options
    preprocess_group = parser.add_argument_group('Preprocessing Options')
    preprocess_group.add_argument('--preprocessing-mode', type=str,
                                choices=['sparse', 'dense_interpolated', 'dense_daily_recalc'],
                                default='sparse',
                                help='Preprocessing mode (default: sparse)')
    preprocess_group.add_argument('--weekly-codes', nargs='+',
                                help='Weekly codes to preprocess')
    preprocess_group.add_argument('--force-reprocess', action='store_true',
                                help='Force reprocessing of existing data')
    
    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help='Override compute device')
    
    return parser.parse_args()


def handle_portfolio_commands(args, config):
    """Handle portfolio-related commands"""
    portfolio_mgr = PortfolioManager(config)
    
    if args.list_portfolios:
        templates = portfolio_mgr.list_available_templates()
        print("Available Portfolio Templates:")
        print("=" * 40)
        
        for template_name in templates:
            info = portfolio_mgr.get_template_info(template_name)
            if info:
                availability = info['data_availability']
                print(f"• {template_name}")
                print(f"  Description: {info['description']}")
                print(f"  Positions: {info['num_positions']}")
                print(f"  Data Available: {availability['available_positions']}/{availability['total_positions']} "
                      f"({availability['availability_percentage']:.1f}%)")
                print()
        
        return True  # Exit after listing
    
    if args.portfolio_info:
        info = portfolio_mgr.get_template_info(args.portfolio_info)
        if info:
            print(f"Portfolio Template: {args.portfolio_info}")
            print("=" * 50)
            print(f"Description: {info['description']}")
            print(f"Number of Positions: {info['num_positions']}")
            print()
            
            print("Positions:")
            for option_key, position_size in info['positions'].items():
                print(f"  {option_key}: {position_size}")
            print()
            
            availability = info['data_availability']
            print(f"Data Availability: {availability['available_positions']}/{availability['total_positions']} "
                  f"({availability['availability_percentage']:.1f}%)")
            
            if availability['missing_data']:
                print("\nMissing Data:")
                for missing in availability['missing_data']:
                    print(f"  • {missing}")
                    
            if availability['available_data']:
                print("\nAvailable Data:")
                for available in availability['available_data']:
                    print(f"  • {available}")
        else:
            print(f"Portfolio template '{args.portfolio_info}' not found")
            print(f"Available templates: {portfolio_mgr.list_available_templates()}")
        
        return True  # Exit after showing info
    
    return False  # Continue execution


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
logger = get_logger(__name__)
    
    try:
        # Create configuration
        cli_args = vars(args)
        config = create_config(cli_args=cli_args)
        
        # Handle portfolio commands (may exit)
        if handle_portfolio_commands(args, config):
            return 0
        
        # Create orchestrator
        orchestrator = ApplicationOrchestrator(config)
        
        # Determine workflow type and execute
        if args.eval_only:
            logger.info("Starting evaluation workflow")
            result = orchestrator.execute_workflow(
                WorkflowType.EVALUATION,
                model_path=args.model,
                eval_samples=args.eval_samples
            )
        
        elif args.validate:
            logger.info("Starting validation workflow")
            result = orchestrator.execute_workflow(
                WorkflowType.VALIDATION,
                validation_type=args.validation_type
            )
        
        elif args.preprocess:
            logger.info("Starting preprocessing workflow")
            result = orchestrator.execute_workflow(
                WorkflowType.PREPROCESSING,
                weekly_codes=args.weekly_codes,
                force_reprocess=args.force_reprocess,
                mode=args.preprocessing_mode
            )
        
        else:  # Training mode (default)
            logger.info("Starting training workflow")
            result = orchestrator.execute_workflow(
                WorkflowType.TRAINING,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                portfolio=args.portfolio,
                resume=args.resume,
                checkpoint_interval=args.checkpoint_interval
            )
        
        # Report results
        if result.succeeded:
            logger.info(f"Workflow completed successfully in {result.execution_time:.2f}s")
            
            # Print key results
            if result.results:
                print("\nResults Summary:")
                print("=" * 30)
                
                if 'training_history' in result.results:
                    history = result.results['training_history']
                    if history:
                        final_metrics = history[-1]
                        print(f"Final Training Loss: {final_metrics['train']['total_loss']:.4f}")
                        if 'validation' in final_metrics and final_metrics['validation']:
                            print(f"Final Validation Loss: {final_metrics['validation']['total_loss']:.4f}")
                        print(f"Best Epoch: {result.results.get('best_epoch', 'N/A')}")
                
                if 'evaluation_metrics' in result.results:
                    eval_metrics = result.results['evaluation_metrics']
                    print(f"Evaluation completed with {len(eval_metrics)} metric groups")
                
                if 'processed_options' in result.results:
                    processed = result.results['processed_options']
                    failed = result.results['failed_options']
                    print(f"Preprocessing: {len(processed)} processed, {len(failed)} failed")
            
            return 0
        
        else:
            logger.error(f"Workflow failed: {result.error}")
            if args.verbose and result.metadata.get('traceback'):
                print("\nFull traceback:")
                print(result.metadata['traceback'])
            return 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())