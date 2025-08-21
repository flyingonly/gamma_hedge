# -*- coding: utf-8 -*-
"""
Run Single Option Delta Hedge Experiment
========================================

This script runs a complete training and evaluation pipeline for the optimal
execution policy on a single option portfolio using the delta hedge dataset.
"""

import torch
import numpy as np
import sys
import os
import argparse

# Add project root to path to allow module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import create_delta_data_loader, DeltaHedgeDataset
from models.policy_network import PolicyNetwork
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.visualization import plot_training_history, plot_results, plot_execution_pattern
from utils.config import initialize_all_configs

def run_experiment():
    """
    Main function to run the single option hedging experiment.
    """
    # --- 0. Initialize Configuration System ---
    # This is a crucial step to register all config types before they are used.
    initialize_all_configs()
    
    print("--- Starting Single Option Delta Hedge Experiment ---")

    # --- 1. Configuration ---
    # Define the portfolio: a single long position in the option with the most data.
    # This was identified as '3CQ5/PUT_114.0'.
    portfolio = {'3CQ5/PUT_114.0': 1.0}
    
    # Training parameters
    n_epochs = 10  # A short training run for demonstration
    batch_size = 32
    learning_rate = 1e-3
    sequence_length = 100 # As defined in default config
    resume_training = False  # Don't resume from checkpoints for this demo

    print(f"\nPortfolio: {portfolio}")
    print(f"Training for {n_epochs} epochs with batch size {batch_size}.")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 2. Data Loading ---
    print("\nLoading data...")
    try:
        # Use the dataset to determine the number of underlying assets
        temp_dataset = DeltaHedgeDataset(option_positions=portfolio)
        n_assets = temp_dataset.n_assets
        
        print(f"Detected {n_assets} underlying asset(s) from option portfolio")

        # Create data loaders for training and validation
        train_loader = create_delta_data_loader(
            batch_size=batch_size,
            option_positions=portfolio
        )
        val_loader = create_delta_data_loader(
            batch_size=batch_size,
            option_positions=portfolio
        )
    except Exception as e:
        print(f"\nError: Failed to load data. Please ensure data is processed correctly.")
        print(f"Details: {e}")
        return

    # --- 3. Model Initialization ---
    print("\nInitializing policy network...")
    input_dim = n_assets * 3  # prev_holding, current_holding, price
    policy_net = PolicyNetwork(input_dim=input_dim)
    
    # --- 4. Training ---
    print("\nStarting training...")
    trainer = Trainer(
        policy_network=policy_net,
        learning_rate=learning_rate,
        device=device
    )
    
    history = trainer.train(
        train_loader,
        n_epochs=n_epochs,
        val_loader=val_loader,
        checkpoint_interval=5, # Save checkpoints less frequently for short runs
        resume=resume_training
    )

    # --- 5. Evaluation ---
    print("\nEvaluating final model...")
    evaluator = Evaluator(policy_net, device=device)

    # Generate a single batch of test data from the validation loader
    test_prices, test_holdings = next(iter(val_loader))
    
    # Compare the learned policy against baseline strategies
    comparison = evaluator.compare_policies(test_prices, test_holdings)

    print("\n--- Strategy Comparison Results ---")
    for strategy, metrics in comparison.items():
        print(f"{strategy:15s}: Avg Cost = {metrics['total_cost']:.4f} (Std: {metrics['cost_std']:.4f})")
    print("------------------------------------")

    # --- 6. Visualization ---
    print("\nGenerating result visualizations...")
    
    # Plot 1: Training and validation loss over epochs
    plot_training_history(history)
    
    # Plot 2: Bar chart comparing total costs of different strategies
    plot_results(comparison)
    
    # Plot 3: Execution pattern of the learned policy on a sample trajectory
    # We use the first sample from the test batch
    plot_execution_pattern(policy_net, test_prices[0], test_holdings[0])
    
    print("\nExperiment finished. Check for plot windows.")
    print("You can find the saved plots in the project root directory.")

if __name__ == "__main__":
    run_experiment()
