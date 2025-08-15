import torch
import numpy as np
import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.market_simulator import MarketConfig, MarketSimulator
from data.data_loader import create_data_loader
from models.policy_network import PolicyNetwork
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.config import get_config
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
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get configuration
    config = get_config()
    
    # Override epochs if specified
    if args.epochs is not None:
        config['training'].n_epochs = args.epochs
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Calculate input dimension
    n_assets = config['market'].n_assets
    input_dim = n_assets * 3  # prev_holding, current_holding, price
    
    # Create model
    print("\nInitializing model...")
    policy_net = PolicyNetwork(
        input_dim=input_dim,
        hidden_dims=config['model'].hidden_dims,
        dropout_rate=config['model'].dropout_rate
    )
    
    # Create trainer
    trainer = Trainer(
        policy_network=policy_net,
        learning_rate=config['training'].learning_rate,
        device=device
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
    test_simulator = MarketSimulator(config['market'])
    test_prices, test_holdings = test_simulator.generate_batch(1)
    
    # Compare strategies
    comparison = evaluator.compare_policies(test_prices, test_holdings)
    
    print("\nStrategy Comparison:")
    for strategy, metrics in comparison.items():
        print(f"{strategy:15s}: Cost = {metrics['total_cost']:.2f} ± {metrics['cost_std']:.2f}")
    
    # Visualize results
    plot_results(comparison)
    plot_execution_pattern(policy_net, test_prices[0], test_holdings[0])
    
    # Save final model (if not eval-only mode)
    if not args.eval_only:
        print("\nSaving final model...")
        final_save_path = 'optimal_execution_model.pth'
        torch.save({
            'model_state_dict': policy_net.state_dict(),
            'config': config,
            'history': history
        }, final_save_path)
        print(f"Final model saved to: {final_save_path}")
    
    print("\nComplete!")

if __name__ == "__main__":
    main()