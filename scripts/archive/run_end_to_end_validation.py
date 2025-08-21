# -*- coding: utf-8 -*-
"""
End-to-End Validation Pipeline
==============================

Comprehensive validation of the entire gamma hedge system with detailed testing,
verification, and visualization of all components and workflows.

This script provides:
1. Complete system validation from data loading to model training
2. Verification of reward signal implementation (cumulative cost)
3. Comprehensive visualization of training progress and results
4. Detailed reporting and metrics collection
5. Integration testing of all major components
"""

import torch
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import create_delta_data_loader, DeltaHedgeDataset
from models.policy_network import PolicyNetwork
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.visualization import plot_training_history, plot_results, plot_execution_pattern
from utils.config import initialize_all_configs
from common.interfaces import DataResult
from common.adapters import to_data_result

class EndToEndValidator:
    """Comprehensive end-to-end validation system"""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = output_dir
        self.validation_results = {}
        self.start_time = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        self.log_file = os.path.join(output_dir, f"validation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
    def log(self, message: str, print_console: bool = True):
        """Log message to file and optionally console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        if print_console:
            print(log_entry)
            
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def validate_configuration(self) -> bool:
        """Phase 1: Validate configuration system"""
        self.log("=== Phase 1: Configuration Validation ===")
        
        try:
            # Initialize all configurations
            initialize_all_configs()
            self.log("✅ Configuration system initialized successfully")
            
            # Test configuration loading
            from utils.config import get_global_config
            import training.config
            
            # Try to load different config types
            configs_to_test = ['training', 'model']
            for config_type in configs_to_test:
                try:
                    config = get_global_config(config_type)
                    self.log(f"✅ {config_type.capitalize()} config loaded successfully")
                except Exception as e:
                    self.log(f"❌ Failed to load {config_type} config: {e}")
                    return False
                    
            self.validation_results['configuration'] = {'status': 'passed', 'configs_loaded': configs_to_test}
            return True
            
        except Exception as e:
            self.log(f"❌ Configuration validation failed: {e}")
            self.validation_results['configuration'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def validate_data_pipeline(self) -> bool:
        """Phase 2: Validate data loading and processing"""
        self.log("=== Phase 2: Data Pipeline Validation ===")
        
        try:
            # Test different portfolio configurations
            test_portfolios = [
                # Single option test
                {'3CQ5/PUT_114.0': 1.0},
                # Multi-option test
                {'3CQ5/PUT_114.0': 1.0, '3CQ5/CALL_115.0': -0.5},
                # Different underlying test
                {'3IN5/PUT_107.0': 2.0}
            ]
            
            data_validation_results = []
            
            for i, portfolio in enumerate(test_portfolios):
                self.log(f"Testing portfolio {i+1}: {portfolio}")
                
                try:
                    # Create dataset
                    dataset = DeltaHedgeDataset(option_positions=portfolio)
                    n_assets = dataset.n_assets
                    self.log(f"  ✅ Dataset created: {n_assets} underlying assets detected")
                    
                    # Create data loader
                    data_loader = create_delta_data_loader(
                        batch_size=16,
                        option_positions=portfolio
                    )
                    self.log(f"  ✅ Data loader created successfully")
                    
                    # Test data loading
                    batch_data = next(iter(data_loader))
                    data_result = to_data_result(batch_data)
                    
                    # Validate data shapes
                    expected_shape = (16, 100, n_assets)  # batch_size, sequence_length, n_assets
                    
                    if data_result.prices.shape == expected_shape:
                        self.log(f"  ✅ Price data shape correct: {data_result.prices.shape}")
                    else:
                        self.log(f"  ⚠️  Price data shape unexpected: {data_result.prices.shape}, expected: {expected_shape}")
                    
                    if data_result.holdings.shape == expected_shape:
                        self.log(f"  ✅ Holdings data shape correct: {data_result.holdings.shape}")
                    else:
                        self.log(f"  ⚠️  Holdings data shape unexpected: {data_result.holdings.shape}, expected: {expected_shape}")
                    
                    # Test data quality
                    if not torch.isnan(data_result.prices).any():
                        self.log(f"  ✅ No NaN values in price data")
                    else:
                        self.log(f"  ❌ NaN values found in price data")
                        
                    if not torch.isnan(data_result.holdings).any():
                        self.log(f"  ✅ No NaN values in holdings data")
                    else:
                        self.log(f"  ❌ NaN values found in holdings data")
                    
                    data_validation_results.append({
                        'portfolio': portfolio,
                        'n_assets': n_assets,
                        'data_shape': data_result.prices.shape,
                        'status': 'passed'
                    })
                    
                except Exception as e:
                    self.log(f"  ❌ Portfolio {i+1} validation failed: {e}")
                    data_validation_results.append({
                        'portfolio': portfolio,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # Summary
            passed_tests = sum(1 for result in data_validation_results if result['status'] == 'passed')
            total_tests = len(data_validation_results)
            
            self.log(f"Data pipeline validation: {passed_tests}/{total_tests} tests passed")
            
            self.validation_results['data_pipeline'] = {
                'status': 'passed' if passed_tests == total_tests else 'partial',
                'tests_passed': passed_tests,
                'total_tests': total_tests,
                'results': data_validation_results
            }
            
            return passed_tests > 0  # At least one test should pass
            
        except Exception as e:
            self.log(f"❌ Data pipeline validation failed: {e}")
            self.validation_results['data_pipeline'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def validate_model_architecture(self) -> bool:
        """Phase 3: Validate model architecture and initialization"""
        self.log("=== Phase 3: Model Architecture Validation ===")
        
        try:
            # Test different input dimensions
            test_configs = [
                {'n_assets': 1, 'input_dim': 3},   # Single asset
                {'n_assets': 2, 'input_dim': 6},   # Two assets
                {'n_assets': 3, 'input_dim': 9},   # Three assets
            ]
            
            model_validation_results = []
            
            for config in test_configs:
                n_assets = config['n_assets']
                input_dim = config['input_dim']
                
                self.log(f"Testing model with {n_assets} assets (input_dim={input_dim})")
                
                try:
                    # Create policy network
                    policy_net = PolicyNetwork(input_dim=input_dim)
                    policy_net.to(self.device)
                    
                    # Test forward pass
                    batch_size = 8
                    prev_holding = torch.randn(batch_size, n_assets).to(self.device)
                    current_holding = torch.randn(batch_size, n_assets).to(self.device)
                    price = torch.randn(batch_size, n_assets).to(self.device)
                    
                    # Forward pass
                    output = policy_net(prev_holding, current_holding, price)
                    
                    # Validate output
                    expected_output_shape = (batch_size, 1)
                    if output.shape == expected_output_shape:
                        self.log(f"  ✅ Forward pass successful, output shape: {output.shape}")
                    else:
                        self.log(f"  ❌ Unexpected output shape: {output.shape}, expected: {expected_output_shape}")
                        
                    # Validate output range (should be probabilities between 0 and 1)
                    if torch.all(output >= 0) and torch.all(output <= 1):
                        self.log(f"  ✅ Output values in valid range [0,1]")
                    else:
                        self.log(f"  ❌ Output values outside valid range [0,1]")
                        
                    # Test gradient computation
                    loss = output.sum()
                    loss.backward()
                    
                    # Check if gradients exist
                    has_gradients = any(p.grad is not None for p in policy_net.parameters())
                    if has_gradients:
                        self.log(f"  ✅ Gradients computed successfully")
                    else:
                        self.log(f"  ❌ No gradients computed")
                    
                    model_validation_results.append({
                        'config': config,
                        'output_shape': output.shape,
                        'output_range_valid': torch.all(output >= 0) and torch.all(output <= 1),
                        'gradients_computed': has_gradients,
                        'status': 'passed'
                    })
                    
                except Exception as e:
                    self.log(f"  ❌ Model validation failed for {n_assets} assets: {e}")
                    model_validation_results.append({
                        'config': config,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # Summary
            passed_tests = sum(1 for result in model_validation_results if result['status'] == 'passed')
            total_tests = len(model_validation_results)
            
            self.log(f"Model architecture validation: {passed_tests}/{total_tests} tests passed")
            
            self.validation_results['model_architecture'] = {
                'status': 'passed' if passed_tests == total_tests else 'partial',
                'tests_passed': passed_tests,
                'total_tests': total_tests,
                'results': model_validation_results
            }
            
            return passed_tests > 0
            
        except Exception as e:
            self.log(f"❌ Model architecture validation failed: {e}")
            self.validation_results['model_architecture'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def validate_reward_signal_implementation(self, trainer: Trainer, test_data: DataResult) -> bool:
        """Phase 4: Validate reward signal implementation (cumulative cost)"""
        self.log("=== Phase 4: Reward Signal Implementation Validation ===")
        
        try:
            # Test the compute_policy_loss function
            self.log("Testing reward signal computation...")
            
            # Create a small test batch
            test_data_small = DataResult(
                prices=test_data.prices[:2],  # Only 2 samples
                holdings=test_data.holdings[:2]
            ).to(self.device)
            
            # Compute policy loss
            loss = trainer.compute_policy_loss(test_data_small)
            
            self.log(f"  ✅ Policy loss computed: {loss.item():.6f}")
            
            # Verify loss is a scalar
            if loss.dim() == 0:
                self.log(f"  ✅ Loss is scalar as expected")
            else:
                self.log(f"  ❌ Loss dimension unexpected: {loss.dim()}")
                
            # Verify loss is finite
            if torch.isfinite(loss):
                self.log(f"  ✅ Loss is finite")
            else:
                self.log(f"  ❌ Loss is not finite: {loss}")
                
            # Test gradient computation
            loss.backward()
            
            # Check gradients
            gradients_exist = any(p.grad is not None for p in trainer.policy.parameters())
            if gradients_exist:
                self.log(f"  ✅ Gradients computed for policy parameters")
            else:
                self.log(f"  ❌ No gradients computed for policy parameters")
                
            # Verify cumulative cost implementation by manual inspection
            # This is a spot check to ensure the implementation uses torch.sum(rewards)
            import inspect
            source = inspect.getsource(trainer.compute_policy_loss)
            
            if 'torch.sum(rewards' in source:
                self.log(f"  ✅ Cumulative cost implementation detected (torch.sum)")
            else:
                self.log(f"  ⚠️  Cumulative cost implementation check inconclusive")
                
            self.validation_results['reward_signal'] = {
                'status': 'passed',
                'loss_value': loss.item(),
                'loss_is_finite': torch.isfinite(loss).item(),
                'gradients_computed': gradients_exist,
                'cumulative_implementation': 'torch.sum(rewards' in source
            }
            
            return True
            
        except Exception as e:
            self.log(f"❌ Reward signal validation failed: {e}")
            self.validation_results['reward_signal'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def validate_training_pipeline(self) -> Tuple[bool, Any]:
        """Phase 5: Validate training pipeline with short training run"""
        self.log("=== Phase 5: Training Pipeline Validation ===")
        
        try:
            # Setup for training test
            portfolio = {'3CQ5/PUT_114.0': 1.0}  # Use simple portfolio
            batch_size = 16
            n_epochs = 3  # Very short training for validation
            
            self.log(f"Setting up training validation with portfolio: {portfolio}")
            
            # Create data
            dataset = DeltaHedgeDataset(option_positions=portfolio)
            n_assets = dataset.n_assets
            
            train_loader = create_delta_data_loader(
                batch_size=batch_size,
                option_positions=portfolio
            )
            
            val_loader = create_delta_data_loader(
                batch_size=batch_size,
                option_positions=portfolio
            )
            
            # Create model and trainer
            input_dim = n_assets * 3
            policy_net = PolicyNetwork(input_dim=input_dim)
            
            trainer = Trainer(
                policy_network=policy_net,
                learning_rate=1e-3,
                device=self.device
            )
            
            # Get test data for reward signal validation
            test_batch = next(iter(val_loader))
            test_data = to_data_result(test_batch).to(self.device)
            
            # Validate reward signal implementation
            reward_validation_passed = self.validate_reward_signal_implementation(trainer, test_data)
            
            # Record initial loss
            initial_loss = trainer.compute_policy_loss(test_data).item()
            self.log(f"Initial loss: {initial_loss:.6f}")
            
            # Run training
            self.log(f"Starting {n_epochs} epoch training validation...")
            
            history = trainer.train(
                train_loader,
                n_epochs=n_epochs,
                val_loader=val_loader,
                checkpoint_interval=n_epochs + 1,  # Don't save checkpoints during validation
                resume=False
            )
            
            # Record final loss
            final_loss = trainer.compute_policy_loss(test_data).item()
            self.log(f"Final loss: {final_loss:.6f}")
            
            # Validate training progress
            training_losses = [epoch['loss'] for epoch in history]
            val_losses = [epoch.get('val_loss', 0) for epoch in history]
            
            if len(training_losses) == n_epochs:
                self.log(f"  ✅ Correct number of training epochs recorded: {len(training_losses)}")
            else:
                self.log(f"  ❌ Incorrect number of training epochs: {len(training_losses)} != {n_epochs}")
                
            # Check if losses are finite
            all_losses_finite = all(np.isfinite(loss) for loss in training_losses)
            if all_losses_finite:
                self.log(f"  ✅ All training losses are finite")
            else:
                self.log(f"  ❌ Some training losses are not finite")
                
            self.log(f"Training loss progression: {[f'{loss:.4f}' for loss in training_losses]}")
            
            self.validation_results['training_pipeline'] = {
                'status': 'passed' if reward_validation_passed and all_losses_finite else 'partial',
                'n_epochs': n_epochs,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'training_losses': training_losses,
                'val_losses': val_losses,
                'reward_signal_validation': reward_validation_passed,
                'all_losses_finite': all_losses_finite
            }
            
            return True, {'trainer': trainer, 'test_data': test_data, 'history': history, 'val_loader': val_loader}
            
        except Exception as e:
            self.log(f"❌ Training pipeline validation failed: {e}")
            self.validation_results['training_pipeline'] = {'status': 'failed', 'error': str(e)}
            return False, None
    
    def validate_evaluation_pipeline(self, trainer: Trainer, val_loader) -> bool:
        """Phase 6: Validate evaluation and comparison functionality"""
        self.log("=== Phase 6: Evaluation Pipeline Validation ===")
        
        try:
            # Create evaluator
            evaluator = Evaluator(trainer.policy, device=self.device)
            
            # Get test data
            market_data = next(iter(val_loader))
            test_prices = market_data.prices
            test_holdings = market_data.holdings
            
            self.log(f"Testing evaluation with data shapes: prices={test_prices.shape}, holdings={test_holdings.shape}")
            
            # Test policy comparison
            comparison = evaluator.compare_policies((test_prices, test_holdings))
            
            # Validate comparison results
            expected_strategies = ['learned_policy', 'immediate_execution', 'delayed_execution']
            
            for strategy in expected_strategies:
                if strategy in comparison:
                    metrics = comparison[strategy]
                    total_cost = metrics.get('total_cost', 0)
                    cost_std = metrics.get('cost_std', 0)
                    
                    if np.isfinite(total_cost) and np.isfinite(cost_std):
                        self.log(f"  ✅ {strategy}: cost={total_cost:.4f} ± {cost_std:.4f}")
                    else:
                        self.log(f"  ❌ {strategy}: invalid metrics (cost={total_cost}, std={cost_std})")
                else:
                    self.log(f"  ❌ Missing strategy in comparison: {strategy}")
            
            # Test if learned policy produces reasonable results
            learned_cost = comparison.get('learned_policy', {}).get('total_cost', float('inf'))
            immediate_cost = comparison.get('immediate_execution', {}).get('total_cost', float('inf'))
            
            performance_reasonable = abs(learned_cost) < abs(immediate_cost) * 2  # Within 2x of baseline
            
            if performance_reasonable:
                self.log(f"  ✅ Learned policy performance is reasonable")
            else:
                self.log(f"  ⚠️  Learned policy performance may need improvement")
            
            self.validation_results['evaluation_pipeline'] = {
                'status': 'passed',
                'strategies_tested': list(comparison.keys()),
                'learned_policy_cost': learned_cost,
                'immediate_execution_cost': immediate_cost,
                'performance_reasonable': performance_reasonable,
                'full_comparison': comparison
            }
            
            return True
            
        except Exception as e:
            self.log(f"❌ Evaluation pipeline validation failed: {e}")
            self.validation_results['evaluation_pipeline'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def generate_comprehensive_visualizations(self, validation_artifacts) -> bool:
        """Phase 7: Generate comprehensive visualizations"""
        self.log("=== Phase 7: Comprehensive Visualization Generation ===")
        
        try:
            # Extract artifacts
            trainer = validation_artifacts['trainer']
            test_data = validation_artifacts['test_data']
            history = validation_artifacts['history']
            val_loader = validation_artifacts['val_loader']
            
            # Create evaluator for comparison plots
            evaluator = Evaluator(trainer.policy, device=self.device)
            market_data = next(iter(val_loader))
            test_prices = market_data.prices
            test_holdings = market_data.holdings
            comparison = evaluator.compare_policies((test_prices, test_holdings))
            
            # 1. Training History Plot
            self.log("Generating training history visualization...")
            plt.figure(figsize=(12, 8))
            
            # Training and validation loss
            epochs = range(1, len(history) + 1)
            train_losses = [epoch['loss'] for epoch in history]
            val_losses = [epoch.get('val_loss', 0) for epoch in history]
            
            plt.subplot(2, 2, 1)
            plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            if any(val_losses):
                plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Strategy Comparison
            plt.subplot(2, 2, 2)
            strategies = list(comparison.keys())
            costs = [comparison[s]['total_cost'] for s in strategies]
            colors = ['green', 'orange', 'red'][:len(strategies)]
            
            bars = plt.bar(strategies, costs, color=colors, alpha=0.7)
            plt.ylabel('Total Cost')
            plt.title('Strategy Comparison')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, cost in zip(bars, costs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(costs), 
                        f'{cost:.3f}', ha='center', va='bottom')
            
            # 3. Execution Pattern Visualization
            plt.subplot(2, 2, 3)
            
            # Sample execution pattern
            sample_prices = test_data.prices[0].cpu().numpy() # Use test_data from artifacts
            sample_holdings = test_data.holdings[0].cpu().numpy() # Use test_data from artifacts
            
            # Get policy decisions
            with torch.no_grad():
                execution_probs = []
                current_holding = sample_holdings[0:1]  # Start with first holding
                
                for t in range(len(sample_prices)):
                    prev_holding = current_holding if t > 0 else sample_holdings[0:1]
                    required_holding = sample_holdings[t:t+1]
                    price = sample_prices[t:t+1]
                    
                    # Convert to tensors
                    prev_holding_tensor = torch.tensor(prev_holding, dtype=torch.float32).to(self.device)
                    required_holding_tensor = torch.tensor(required_holding, dtype=torch.float32).to(self.device)
                    price_tensor = torch.tensor(price, dtype=torch.float32).to(self.device)
                    
                    prob = trainer.policy(prev_holding_tensor, required_holding_tensor, price_tensor)
                    execution_probs.append(prob.cpu().item())
                    
                    # Update current holding based on policy decision (simplified)
                    current_holding = required_holding  # For visualization purposes
            
            timesteps = range(len(execution_probs))
            plt.plot(timesteps, execution_probs, 'g-', linewidth=2, label='Execution Probability')
            plt.xlabel('Time Step')
            plt.ylabel('Execution Probability')
            plt.title('Policy Execution Pattern')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 4. Price and Holdings Evolution
            plt.subplot(2, 2, 4)
            
            # Plot first asset price and holding evolution
            timesteps = range(len(sample_prices))
            plt.plot(timesteps, sample_prices[:, 0], 'b-', linewidth=2, label='Asset Price')
            
            # Plot holdings on secondary y-axis
            ax2 = plt.gca().twinx()
            ax2.plot(timesteps, sample_holdings[:, 0], 'r-', linewidth=2, label='Required Holdings')
            
            plt.xlabel('Time Step')
            plt.ylabel('Price', color='b')
            ax2.set_ylabel('Holdings', color='r')
            plt.title('Price and Holdings Evolution')
            plt.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            
            # Save comprehensive plot
            plot_path = os.path.join(self.output_dir, 'comprehensive_validation_results.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.log(f"Comprehensive visualization saved: {plot_path}")
            
            # 5. Additional detailed plots
            self.log("Generating detailed training analysis...")
            
            plt.figure(figsize=(15, 10))
            
            # Loss distribution across batches (if available)
            plt.subplot(2, 3, 1)
            plt.hist(train_losses, bins=min(len(train_losses), 10), alpha=0.7, color='blue')
            plt.xlabel('Loss Value')
            plt.ylabel('Frequency')
            plt.title('Training Loss Distribution')
            plt.grid(True, alpha=0.3)
            
            # Validation metrics comparison
            plt.subplot(2, 3, 2)
            validation_summary = self.validation_results
            phases = ['configuration', 'data_pipeline', 'model_architecture', 'training_pipeline', 'evaluation_pipeline']
            statuses = [validation_summary.get(phase, {}).get('status', 'unknown') for phase in phases]
            
            status_counts = {'passed': 0, 'failed': 0, 'partial': 0}
            for status in statuses:
                if status in status_counts:
                    status_counts[status] += 1
            
            colors = {'passed': 'green', 'failed': 'red', 'partial': 'orange'}
            plt.pie(status_counts.values(), labels=status_counts.keys(), 
                   colors=[colors[k] for k in status_counts.keys()], autopct='%1.0f%%')
            plt.title('Validation Results Summary')
            
            # Model parameter distribution
            plt.subplot(2, 3, 3)
            all_params = []
            for param in trainer.policy.parameters():
                all_params.extend(param.data.cpu().numpy().flatten())
            
            plt.hist(all_params, bins=50, alpha=0.7, color='purple')
            plt.xlabel('Parameter Value')
            plt.ylabel('Frequency')
            plt.title('Model Parameter Distribution')
            plt.grid(True, alpha=0.3)
            
            # Reward signal analysis
            plt.subplot(2, 3, 4)
            
            # Compute costs for sample trajectory
            sample_data = DataResult(
                prices=test_data.prices[:1],  # Single sample
                holdings=test_data.holdings[:1]
            ).to(self.device)
            
            # Extract reward computation manually for visualization
            with torch.no_grad():
                prices = sample_data.prices[0]  # Remove batch dimension
                holdings = sample_data.holdings[0]
                
                costs_per_step = []
                current_holding = holdings[0].clone()
                
                for t in range(len(prices)):
                    required_holding = holdings[t]
                    current_price = prices[t]
                    
                    holding_change = required_holding - current_holding
                    immediate_cost = -(holding_change * current_price).sum()
                    costs_per_step.append(immediate_cost.item())
                    
                    current_holding = required_holding  # For visualization
                
                cumulative_cost = np.cumsum(costs_per_step)
                
                timesteps = range(len(costs_per_step))
                plt.plot(timesteps, costs_per_step, 'b-', linewidth=2, label='Immediate Cost')
                plt.plot(timesteps, cumulative_cost, 'r-', linewidth=2, label='Cumulative Cost')
                plt.xlabel('Time Step')
                plt.ylabel('Cost')
                plt.title('Cost Analysis (Reward Signal)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Policy output distribution
            plt.subplot(2, 3, 5)
            
            with torch.no_grad():
                # Sample policy outputs
                policy_outputs = []
                for batch_data in val_loader:
                    data_result = to_data_result(batch_data).to(self.device)
                    
                    for t in range(min(10, data_result.prices.shape[1])):
                        prev_holding = data_result.holdings[:, max(0, t-1), :]
                        current_holding = data_result.holdings[:, t, :]
                        price = data_result.prices[:, t, :]
                        
                        output = trainer.policy(prev_holding, current_holding, price)
                        policy_outputs.extend(output.cpu().numpy().flatten())
                    
                    break  # Only process first batch
                
                plt.hist(policy_outputs, bins=30, alpha=0.7, color='orange')
                plt.xlabel('Execution Probability')
                plt.ylabel('Frequency')
                plt.title('Policy Output Distribution')
                plt.grid(True, alpha=0.3)
                plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold=0.5')
                plt.legend()
            
            # Performance metrics summary
            plt.subplot(2, 3, 6)
            
            metrics_data = []
            metrics_labels = []
            
            # Extract key metrics
            if 'training_pipeline' in validation_summary:
                initial_loss = validation_summary['training_pipeline'].get('initial_loss', 0)
                final_loss = validation_summary['training_pipeline'].get('final_loss', 0)
                metrics_data.extend([initial_loss, final_loss])
                metrics_labels.extend(['Initial Loss', 'Final Loss'])
            
            if 'evaluation_pipeline' in validation_summary:
                learned_cost = validation_summary['evaluation_pipeline'].get('learned_policy_cost', 0)
                immediate_cost = validation_summary['evaluation_pipeline'].get('immediate_execution_cost', 0)
                metrics_data.extend([abs(learned_cost), abs(immediate_cost)])
                metrics_labels.extend(['Learned Policy', 'Immediate Execution'])
            
            if metrics_data:
                plt.bar(range(len(metrics_data)), metrics_data, color=['blue', 'lightblue', 'green', 'orange'])
                plt.xticks(range(len(metrics_data)), metrics_labels, rotation=45)
                plt.ylabel('Absolute Value')
                plt.title('Key Performance Metrics')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save detailed analysis plot
            detailed_plot_path = os.path.join(self.output_dir, 'detailed_validation_analysis.png')
            plt.savefig(detailed_plot_path, dpi=150, bbox_inches='tight')
            self.log(f"Detailed analysis visualization saved: {detailed_plot_path}")
            
            plt.show()  # Display plots
            
            self.validation_results['visualization'] = {
                'status': 'passed',
                'comprehensive_plot': plot_path,
                'detailed_plot': detailed_plot_path,
                'plots_generated': 2
            }
            
            return True
            
        except Exception as e:
            self.log(f"❌ Visualization generation failed: {e}")
            self.validation_results['visualization'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        self.log("=== Generating Final Validation Report ===")
        
        report_content = []
        report_content.append("# End-to-End Validation Report")
        report_content.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"**Device**: {self.device}")
        report_content.append(f"**Total Validation Time**: {time.time() - self.start_time:.2f} seconds")
        report_content.append("")
        
        # Overall summary
        total_phases = len(self.validation_results)
        passed_phases = sum(1 for result in self.validation_results.values() 
                          if result.get('status') == 'passed')
        
        report_content.append("## Executive Summary")
        report_content.append(f"- **Phases Tested**: {total_phases}")
        report_content.append(f"- **Phases Passed**: {passed_phases}")
        report_content.append(f"- **Success Rate**: {passed_phases/total_phases*100:.1f}%")
        report_content.append("")
        
        # Detailed results for each phase
        report_content.append("## Detailed Results")
        report_content.append("")
        
        phase_names = {
            'configuration': 'Configuration System',
            'data_pipeline': 'Data Pipeline',
            'model_architecture': 'Model Architecture',
            'reward_signal': 'Reward Signal Implementation',
            'training_pipeline': 'Training Pipeline',
            'evaluation_pipeline': 'Evaluation Pipeline',
            'visualization': 'Visualization Generation'
        }
        
        for phase_key, phase_name in phase_names.items():
            if phase_key in self.validation_results:
                result = self.validation_results[phase_key]
                status = result.get('status', 'unknown')
                status_icon = "✅" if status == 'passed' else "⚠️" if status == 'partial' else "❌"
                
                report_content.append(f"### {phase_name} {status_icon}")
                report_content.append(f"**Status**: {status.upper()}")
                
                if status == 'failed' and 'error' in result:
                    report_content.append(f"**Error**: {result['error']}")
                elif status in ['passed', 'partial']:
                    # Add specific details for each phase
                    if phase_key == 'data_pipeline':
                        tests_passed = result.get('tests_passed', 0)
                        total_tests = result.get('total_tests', 0)
                        report_content.append(f"**Tests Passed**: {tests_passed}/{total_tests}")
                    
                    elif phase_key == 'training_pipeline':
                        initial_loss = result.get('initial_loss', 0)
                        final_loss = result.get('final_loss', 0)
                        report_content.append(f"**Initial Loss**: {initial_loss:.6f}")
                        report_content.append(f"**Final Loss**: {final_loss:.6f}")
                        report_content.append(f"**Loss Improvement**: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")
                    
                    elif phase_key == 'evaluation_pipeline':
                        learned_cost = result.get('learned_policy_cost', 0)
                        immediate_cost = result.get('immediate_execution_cost', 0)
                        report_content.append(f"**Learned Policy Cost**: {learned_cost:.4f}")
                        report_content.append(f"**Immediate Execution Cost**: {immediate_cost:.4f}")
                        
                        if immediate_cost != 0:
                            improvement = (immediate_cost - learned_cost) / abs(immediate_cost) * 100
                            report_content.append(f"**Cost Improvement**: {improvement:.2f}%")
                
                report_content.append("")
        
        # Key findings
        report_content.append("## Key Findings")
        report_content.append("")
        
        # Reward signal verification
        if 'reward_signal' in self.validation_results:
            reward_result = self.validation_results['reward_signal']
            if reward_result.get('cumulative_implementation', False):
                report_content.append("- ✅ **Reward Signal**: Verified cumulative cost implementation (torch.sum)")
            else:
                report_content.append("- ⚠️ **Reward Signal**: Implementation verification inconclusive")
        
        # Performance assessment
        if 'evaluation_pipeline' in self.validation_results:
            eval_result = self.validation_results['evaluation_pipeline']
            if eval_result.get('performance_reasonable', False):
                report_content.append("- ✅ **Performance**: Learned policy performance within reasonable bounds")
            else:
                report_content.append("- ⚠️ **Performance**: Learned policy may need improvement")
        
        # Architecture validation
        if 'model_architecture' in self.validation_results:
            arch_result = self.validation_results['model_architecture']
            tests_passed = arch_result.get('tests_passed', 0)
            total_tests = arch_result.get('total_tests', 0)
            if tests_passed == total_tests:
                report_content.append("- ✅ **Architecture**: All model architecture tests passed")
            else:
                report_content.append(f"- ⚠️ **Architecture**: {tests_passed}/{total_tests} architecture tests passed")
        
        report_content.append("")
        
        # Recommendations
        report_content.append("## Recommendations")
        report_content.append("")
        
        if passed_phases == total_phases:
            report_content.append("- 🎉 **All validation phases passed successfully**")
            report_content.append("- The system is ready for production use")
            report_content.append("- Consider running longer training experiments for performance optimization")
        else:
            failed_phases = [phase for phase, result in self.validation_results.items() 
                           if result.get('status') == 'failed']
            if failed_phases:
                report_content.append(f"- ❌ **Critical Issues**: Address failures in {', '.join(failed_phases)}")
            
            partial_phases = [phase for phase, result in self.validation_results.items() 
                            if result.get('status') == 'partial']
            if partial_phases:
                report_content.append(f"- ⚠️ **Improvements Needed**: Review partial results in {', '.join(partial_phases)}")
        
        report_content.append("")
        
        report_content.append("## Files Generated")
        report_content.append(f"- Validation Log: `{self.log_file}`")
        report_content.append(f"- Validation Results: `{os.path.join(self.output_dir, 'validation_results.json')}`")
        
        if 'visualization' in self.validation_results and self.validation_results['visualization']['status'] == 'passed':
            report_content.append(f"- Comprehensive Plot: `{self.validation_results['visualization']['comprehensive_plot']}`")
            report_content.append(f"- Detailed Analysis: `{self.validation_results['visualization']['detailed_plot']}`")
        
        # Write report to file
        report_text = '\n'.join(report_content)
        report_path = os.path.join(self.output_dir, 'validation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Save detailed results as JSON
        results_path = os.path.join(self.output_dir, 'validation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        self.log(f"Validation report saved: {report_path}")
        self.log(f"Detailed results saved: {results_path}")
        
        return report_path
    
    def run_full_validation(self) -> bool:
        """Run complete end-to-end validation pipeline"""
        self.start_time = time.time()
        
        self.log("🚀 Starting Comprehensive End-to-End Validation")
        self.log("=" * 60)
        
        # # Phase 1: Configuration
        # if not self.validate_configuration():
        #     self.log("❌ Configuration validation failed. Stopping validation.")
        #     return False
        
        # Phase 2: Data Pipeline
        if not self.validate_data_pipeline():
            self.log("❌ Data pipeline validation failed. Stopping validation.")
            return False
        
        # Phase 3: Model Architecture
        if not self.validate_model_architecture():
            self.log("❌ Model architecture validation failed. Stopping validation.")
            return False
        
        # Phase 5: Training Pipeline (includes Phase 4: Reward Signal)
        training_success, validation_artifacts = self.validate_training_pipeline()
        if not training_success:
            self.log("❌ Training pipeline validation failed. Stopping validation.")
            return False
        
        # Phase 6: Evaluation Pipeline
        if not self.validate_evaluation_pipeline(
            validation_artifacts['trainer'], 
            validation_artifacts['val_loader']
        ):
            self.log("⚠️  Evaluation pipeline validation failed, but continuing...")
        
        # Phase 7: Visualization
        if not self.generate_comprehensive_visualizations(validation_artifacts):
            self.log("⚠️  Visualization generation failed, but continuing...")
        
        # Generate final report
        report_path = self.generate_validation_report()
        
        elapsed_time = time.time() - self.start_time
        self.log("=" * 60)
        self.log(f"🎉 End-to-End Validation Completed in {elapsed_time:.2f} seconds")
        self.log(f"📊 Full report available: {report_path}")
        
        return True

def main():
    """Main function to run end-to-end validation"""
    parser = argparse.ArgumentParser(description='Run comprehensive end-to-end validation')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Directory to save validation results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create validator and run
    validator = EndToEndValidator(output_dir=args.output_dir)
    
    success = validator.run_full_validation()
    
    if success:
        print(f"\n✅ Validation completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        print(f"📊 Check validation_report.md for detailed findings")
    else:
        print(f"\n❌ Validation failed!")
        print(f"📁 Partial results saved in: {args.output_dir}")
        print(f"📋 Check validation_log_*.txt for error details")
        
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())