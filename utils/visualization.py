import matplotlib
matplotlib.set_loglevel('warning')

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import torch

def plot_training_history(history: List[Dict[str, float]]):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training loss
    epochs = range(1, len(history) + 1)
    train_losses = [h['loss'] for h in history]
    train_stds = [h['loss_std'] for h in history]
    
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].fill_between(epochs,
                         np.array(train_losses) - np.array(train_stds),
                         np.array(train_losses) + np.array(train_stds),
                         alpha=0.3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Validation loss if available
    if 'val_loss' in history[0]:
        val_losses = [h['val_loss'] for h in history]
        axes[1].plot(epochs, val_losses, 'r-', label='Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_execution_pattern(policy_network: torch.nn.Module,
                          prices: torch.Tensor,
                          holdings: torch.Tensor):
    """Visualize execution decisions"""
    policy_network.eval()
    
    with torch.no_grad():
        # Get execution probabilities
        n_timesteps = prices.shape[0] if len(prices.shape) == 2 else prices.shape[1]
        exec_probs = []
        
        if len(prices.shape) == 2:
            prices = prices.unsqueeze(0)
            holdings = holdings.unsqueeze(0)
        
        current_holding = holdings[:, 0, :]
        
        for t in range(n_timesteps):
            prob = policy_network(
                current_holding,
                holdings[:, t, :],
                prices[:, t, :]
            )
            exec_probs.append(prob.cpu().numpy())
            
            # Update holding based on probability (for visualization)
            execute = prob > 0.5
            execute_mask = execute.expand_as(current_holding)
            current_holding = torch.where(
                execute_mask,
                holdings[:, t, :],
                current_holding
            )
    
    exec_probs = np.array(exec_probs).squeeze()
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Prices
    timesteps = range(n_timesteps)
    axes[0].plot(timesteps, prices[0, :, 0].cpu().numpy(), 'b-')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Asset Price Evolution')
    axes[0].grid(True)
    
    # Holdings
    axes[1].plot(timesteps, holdings[0, :, 0].cpu().numpy(), 'g-')
    axes[1].set_ylabel('Holdings')
    axes[1].set_title('Required Holdings')
    axes[1].grid(True)
    
    # Execution probability
    axes[2].bar(timesteps, exec_probs, color='red', alpha=0.6)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Execution Probability')
    axes[2].set_title('Policy Execution Decisions')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_results(comparison_results: Dict[str, Dict[str, float]]):
    """Plot comparison results"""
    strategies = list(comparison_results.keys())
    costs = [comparison_results[s]['total_cost'] for s in strategies]
    stds = [comparison_results[s].get('cost_std', 0) for s in strategies]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, costs, yerr=stds, capsize=5, alpha=0.7)
    
    # Color code bars
    colors = ['green', 'red', 'orange']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Total Cost')
    ax.set_title('Strategy Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies)
    ax.grid(True, axis='y')
    
    # Add value labels on bars
    for i, (cost, std) in enumerate(zip(costs, stds)):
        ax.text(i, cost + std, f'{cost:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()