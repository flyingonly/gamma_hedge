import torch
import numpy as np
import sys
import os
import argparse
from utils.logger import get_logger


# Add project root to path
logger = get_logger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.market_simulator import MarketConfig, MarketSimulator
from data.data_loader import create_data_loader, create_delta_data_loader, check_delta_availability
from data.precomputed_data_loader import PrecomputedGreeksDataset
from models.policy_network import PolicyNetwork
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.config import get_config, create_unified_config
from utils.visualization import plot_training_history, plot_execution_pattern, plot_results

def parse_args():
    parser = argparse.ArgumentParser(description='Optimal Execution Training')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train (overrides config)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate the model without training')
    parser.add_argument('--load-best', action='store_true',
                       help='Load best model for evaluation')
    parser.add_argument('--delta-hedge', action='store_true',
                       help='Use delta hedge mode for training')
    parser.add_argument('--underlying-codes', nargs='+', default=['USU5', 'FVU5'],
                       help='Underlying asset codes for delta hedge')
    parser.add_argument('--weekly-codes', nargs='+', default=['3CN5'],
                       help='Weekly option codes for delta hedge')
    parser.add_argument('--sequence-length', type=int, default=100,
                       help='Sequence length for delta hedge data')
    parser.add_argument('--enable-tracking', action='store_true',
                       help='Enable policy decision tracking for visualization')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations after training/evaluation')
    parser.add_argument('--export-viz', type=str, default=None,
                       help='Export visualizations to specified directory')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get unified configuration
    config = create_unified_config(args)
    
    # Check delta hedge mode
    use_delta_hedge = config['delta_hedge'].enable_delta
    if args.delta_hedge and not use_delta_hedge:
        print("Warning: Delta hedge requested but not available. Falling back to traditional mode.")
    
    print(f"Training mode: {'Delta Hedge' if use_delta_hedge else 'Traditional'}")
    if use_delta_hedge:
        print(f"Underlying codes: {config['delta_hedge'].underlying_codes}")
        print(f"Weekly codes: {config['delta_hedge'].weekly_codes}")
        print(f"Sequence length: {config['delta_hedge'].sequence_length}")
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Calculate input dimension and create model
    if use_delta_hedge:
        # For delta hedging, we always have 1 underlying asset
        n_assets = 1
        print(f"\nInitializing delta hedge model with {n_assets} underlying asset...")
    else:
        n_assets = config['market'].n_assets
        print("\nInitializing traditional model...")
    
    # Calculate input dimension using centralized dimension management
    from core.dimensions import calculate_model_input_dim
    input_dim = calculate_model_input_dim(
        n_assets=n_assets,
        include_history=True,
        include_time=True,
        include_delta=True
    )
    policy_net = PolicyNetwork(
        input_dim=input_dim,
        hidden_dims=config['model'].hidden_dims,
        dropout_rate=config['model'].dropout_rate
    )
    
    # Create trainer
    trainer = Trainer(
        policy_network=policy_net,
        learning_rate=config['training'].learning_rate,
        device=device,
        enable_tracking=args.enable_tracking
    )
    
    # Load checkpoint if specified
    if args.eval_only:
        if args.load_best:
            checkpoint_path = os.path.join('checkpoints', 'best_model.pth')
        elif args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join('checkpoints', 'latest_checkpoint.pth')
        
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            print(f"Loaded model from: {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return
    
    # Training or evaluation mode
    if not args.eval_only:
        # Create data loaders
        print("\nCreating data loaders...")
        if use_delta_hedge:
            train_loader = create_delta_data_loader(
                batch_size=config['training'].batch_size,
                weekly_positions=config['delta_hedge'].weekly_codes
            )
            
            val_loader = create_delta_data_loader(
                batch_size=config['training'].batch_size,
                weekly_positions=config['delta_hedge'].weekly_codes
            )
        else:
            train_loader = create_data_loader(
                config['market'],
                config['training'].n_train_samples,
                config['training'].batch_size
            )
            
            val_loader = create_data_loader(
                config['market'],
                config['training'].n_val_samples,
                config['training'].batch_size
            )
        
        # Train model
        print("\nStarting training...")
        print(f"Checkpoint interval: {args.checkpoint_interval} epochs")
        if args.resume:
            print("Resuming from latest checkpoint...")
        
        history = trainer.train(
            train_loader,
            n_epochs=config['training'].n_epochs,
            val_loader=val_loader,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume
        )
        
        # Plot training history
        print("\nPlotting training history...")
        plot_training_history(history)
    else:
        print("\nSkipping training (eval-only mode)")
        history = trainer.history if hasattr(trainer, 'history') else []
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluator = Evaluator(policy_net, device=device)
    
    # Generate test data
    if use_delta_hedge:
        # For delta hedge mode, create a simple option position from weekly codes
        # Convert weekly codes to option positions format
        weekly_codes = config['delta_hedge'].weekly_codes
        option_positions = {}
        for weekly_code in weekly_codes:
            # Use default ATM call option (this should ideally come from configuration)
            option_positions[f"{weekly_code}/CALL_111.0"] = 1.0
        
        # Generate test data using precomputed dataset
        test_dataset = PrecomputedGreeksDataset(
            portfolio_positions=option_positions,
            sequence_length=50  # Shorter for testing
        )
        
        if len(test_dataset) > 0:
            test_data = test_dataset[0]  # Returns DataResult
            test_prices = test_data.prices.unsqueeze(0)  # Add batch dimension
            test_holdings = test_data.holdings.unsqueeze(0)
        else:
            logger.warning("No precomputed test data available, using simulation fallback")
            test_simulator = MarketSimulator(config['market'])
            test_prices, test_holdings = test_simulator.generate_batch(1)
    else:
        test_simulator = MarketSimulator(config['market'])
        test_prices, test_holdings = test_simulator.generate_batch(1)
    
    # Compare strategies - unified interface
    comparison = evaluator.compare_policies(test_prices, test_holdings)
    
    print("\nStrategy Comparison:")
    for strategy, metrics in comparison.items():
        print(f"{strategy:15s}: Cost = {metrics['total_cost']:.2f} ± {metrics['cost_std']:.2f}")
    
    # Visualize results
    plot_results(comparison)
    plot_execution_pattern(policy_net, test_prices[0], test_holdings[0])
    
    # Generate policy visualization if tracking was enabled
    if args.enable_tracking or args.visualize:
        print("\nGenerating policy behavior visualizations...")
        from utils.policy_visualizer import create_policy_visualizer
        
        visualizer = create_policy_visualizer()
        
        # Generate timeline visualization
        timeline_fig = visualizer.plot_policy_timeline()
        if timeline_fig:
            print("✓ Policy timeline visualization generated")
            
        # Generate hedge analysis
        hedge_fig = visualizer.plot_hedge_analysis()
        if hedge_fig:
            print("✓ Hedge decision analysis generated")
            
        # Generate summary report
        summary_report = visualizer.generate_summary_report()
        print("\nPolicy Behavior Summary:")
        print(summary_report)
        
        # Export visualizations if requested
        if args.export_viz:
            visualizer.export_visualizations(args.export_viz)
            print(f"✓ Visualizations exported to {args.export_viz}/")
    
    # Save final model (if not eval-only mode)
    if not args.eval_only:
        print("\nSaving final model...")
        final_save_path = 'optimal_execution_model.pth'
        save_data = {
            'model_state_dict': policy_net.state_dict(),
            'config': config,
            'history': history,
            'delta_hedge_mode': use_delta_hedge
        }
        
        if use_delta_hedge:
            save_data.update({
                'underlying_codes': config['delta_hedge'].underlying_codes,
                'weekly_codes': config['delta_hedge'].weekly_codes,
                'sequence_length': config['delta_hedge'].sequence_length
            })
        
        torch.save(save_data, final_save_path)
        print(f"Final model saved to: {final_save_path}")
    
    print("\nComplete!")

if __name__ == "__main__":
    main()