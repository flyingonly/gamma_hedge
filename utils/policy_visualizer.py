import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import pandas as pd
from .policy_tracker import PolicyTracker, HedgeDecision

class PolicyVisualizer:
    """
    Visualization tools for Policy behavior analysis
    """
    
    def __init__(self, tracker: PolicyTracker):
        self.tracker = tracker
        self.colors = {
            'hedge': '#FF6B6B',      # Red for hedge actions
            'wait': '#4ECDC4',       # Teal for wait actions  
            'price': '#45B7D1',      # Blue for prices
            'probability': '#96CEB4', # Light green for probabilities
            'cost': '#FFEAA7'        # Yellow for costs
        }
    
    def plot_policy_timeline(self, episode_idx: Optional[int] = None, 
                           show_greeks: bool = True, 
                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create a comprehensive timeline view of policy decisions
        
        Args:
            episode_idx: Episode index to visualize (None for latest)
            show_greeks: Whether to include Greeks in the plot
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        decisions = self.tracker.get_latest_episode() if episode_idx is None else self.tracker.get_episode(episode_idx)
        
        if not decisions:
            print("No decisions to visualize")
            return None
        
        # Prepare data
        timesteps = [d.timestep for d in decisions]
        prices = [d.current_price for d in decisions]
        option_prices = [d.option_price for d in decisions]
        future_prices = [d.future_price for d in decisions]
        execution_probs = [d.execution_probability for d in decisions]
        holdings_current = [d.current_holding for d in decisions]
        holdings_required = [d.required_holding for d in decisions]
        
        # Find hedge points
        hedge_times = [d.timestep for d in decisions if d.action_taken]
        hedge_prices = [d.current_price for d in decisions if d.action_taken]
        hedge_costs = [d.immediate_cost for d in decisions if d.action_taken]
        
        # Create figure
        n_subplots = 5 if show_greeks and any(d.delta is not None for d in decisions) else 4
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        fig.suptitle(f'Policy Decision Timeline - Episode {episode_idx if episode_idx is not None else "Latest"}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Price Evolution with Hedge Points
        ax1 = axes[0]
        ax1.plot(timesteps, prices, color=self.colors['price'], linewidth=2, label='Current Price')
        if option_prices != prices:
            ax1.plot(timesteps, option_prices, color='purple', linewidth=1, alpha=0.7, label='Option Price')
        if future_prices != prices:
            ax1.plot(timesteps, future_prices, color='orange', linewidth=1, alpha=0.7, label='Future Price')
        
        # Mark hedge points
        if hedge_times:
            ax1.scatter(hedge_times, hedge_prices, color=self.colors['hedge'], 
                       s=100, alpha=0.8, marker='v', label='Hedge Actions', zorder=5)
        
        ax1.set_ylabel('Price')
        ax1.set_title('Price Evolution and Hedge Timing')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Execution Probability
        ax2 = axes[1]
        ax2.fill_between(timesteps, execution_probs, color=self.colors['probability'], 
                        alpha=0.6, label='Execution Probability')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
        
        # Mark actual hedge decisions
        if hedge_times:
            hedge_probs = [d.execution_probability for d in decisions if d.action_taken]
            ax2.scatter(hedge_times, hedge_probs, color=self.colors['hedge'], 
                       s=60, alpha=0.8, marker='o', label='Hedge Decisions')
        
        ax2.set_ylabel('Probability')
        ax2.set_title('Policy Execution Probability')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Holdings Comparison
        ax3 = axes[2]
        ax3.plot(timesteps, holdings_current, color='blue', linewidth=2, label='Current Holding')
        ax3.plot(timesteps, holdings_required, color='green', linewidth=2, label='Required Holding')
        ax3.fill_between(timesteps, holdings_current, holdings_required, 
                        alpha=0.3, color='gray', label='Holding Gap')
        
        if hedge_times:
            hedge_current = [d.current_holding for d in decisions if d.action_taken]
            ax3.scatter(hedge_times, hedge_current, color=self.colors['hedge'], 
                       s=60, alpha=0.8, marker='s', label='Hedge Points')
        
        ax3.set_ylabel('Holdings')
        ax3.set_title('Holdings Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cost Analysis
        ax4 = axes[3]
        all_costs = [d.immediate_cost for d in decisions]
        bars = ax4.bar(timesteps, all_costs, color=[self.colors['hedge'] if d.action_taken else self.colors['wait'] 
                                                   for d in decisions], alpha=0.7)
        
        # Highlight hedge costs
        if hedge_times:
            ax4.scatter(hedge_times, hedge_costs, color='darkred', 
                       s=60, alpha=0.9, marker='o', label='Hedge Costs')
        
        ax4.set_ylabel('Immediate Cost')
        ax4.set_title('Cost per Time Step')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Greeks Evolution (if available)
        if show_greeks and n_subplots == 5:
            ax5 = axes[4]
            deltas = [d.delta for d in decisions if d.delta is not None]
            gammas = [d.gamma for d in decisions if d.gamma is not None]
            
            if deltas:
                delta_times = [d.timestep for d in decisions if d.delta is not None]
                ax5.plot(delta_times, deltas, color='blue', linewidth=2, label='Delta')
            
            if gammas:
                gamma_times = [d.timestep for d in decisions if d.gamma is not None]
                ax5_twin = ax5.twinx()
                ax5_twin.plot(gamma_times, gammas, color='red', linewidth=2, label='Gamma')
                ax5_twin.set_ylabel('Gamma', color='red')
                ax5_twin.tick_params(axis='y', labelcolor='red')
            
            ax5.set_ylabel('Delta', color='blue')
            ax5.set_title('Options Greeks Evolution')
            ax5.tick_params(axis='y', labelcolor='blue')
            ax5.grid(True, alpha=0.3)
        
        # Set x-axis label for bottom subplot
        axes[-1].set_xlabel('Time Step')
        
        plt.tight_layout()
        return fig
    
    def plot_hedge_analysis(self, episode_idx: Optional[int] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create detailed analysis of hedge decisions
        
        Args:
            episode_idx: Episode index to analyze
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        decisions = self.tracker.get_latest_episode() if episode_idx is None else self.tracker.get_episode(episode_idx)
        hedge_decisions = [d for d in decisions if d.action_taken]
        
        if not hedge_decisions:
            print("No hedge decisions to analyze")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Hedge Decision Analysis - Episode {episode_idx if episode_idx is not None else "Latest"}',
                     fontsize=14, fontweight='bold')
        
        # 1. Hedge Timing Distribution
        ax1 = axes[0, 0]
        hedge_times = [d.timestep for d in hedge_decisions]
        ax1.hist(hedge_times, bins=min(20, len(hedge_times)), color=self.colors['hedge'], 
                alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Hedge Timing Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Hedge vs Price Relationship
        ax2 = axes[0, 1]
        hedge_prices = [d.current_price for d in hedge_decisions]
        hedge_costs = [d.immediate_cost for d in hedge_decisions]
        
        scatter = ax2.scatter(hedge_prices, hedge_costs, 
                            c=[d.execution_probability for d in hedge_decisions],
                            cmap='viridis', s=60, alpha=0.8, edgecolors='black')
        ax2.set_xlabel('Price at Hedge')
        ax2.set_ylabel('Hedge Cost')
        ax2.set_title('Price vs Cost Relationship')
        plt.colorbar(scatter, ax=ax2, label='Execution Probability')
        ax2.grid(True, alpha=0.3)
        
        # 3. Execution Probability Distribution
        ax3 = axes[1, 0]
        all_probs = [d.execution_probability for d in decisions]
        hedge_probs = [d.execution_probability for d in hedge_decisions]
        wait_probs = [d.execution_probability for d in decisions if not d.action_taken]
        
        ax3.hist([wait_probs, hedge_probs], bins=20, alpha=0.7, 
                color=[self.colors['wait'], self.colors['hedge']], 
                label=['Wait Decisions', 'Hedge Decisions'])
        ax3.set_xlabel('Execution Probability')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Probability Distribution by Action')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Cost Over Time
        ax4 = axes[1, 1]
        all_decisions = sorted(decisions, key=lambda x: x.timestep)
        timesteps = [d.timestep for d in all_decisions]
        cumulative_costs = np.cumsum([d.immediate_cost if d.action_taken else 0 for d in all_decisions])
        
        ax4.plot(timesteps, cumulative_costs, color=self.colors['cost'], linewidth=3)
        ax4.fill_between(timesteps, cumulative_costs, alpha=0.3, color=self.colors['cost'])
        
        # Mark individual hedge points
        hedge_cumulative = []
        running_sum = 0
        for d in all_decisions:
            if d.action_taken:
                running_sum += d.immediate_cost
                hedge_cumulative.append((d.timestep, running_sum))
        
        if hedge_cumulative:
            hedge_times, hedge_cumsum = zip(*hedge_cumulative)
            ax4.scatter(hedge_times, hedge_cumsum, color=self.colors['hedge'], 
                       s=60, alpha=0.8, zorder=5)
        
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Cumulative Cost')
        ax4.set_title('Cumulative Cost Evolution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_strategy_comparison(self, episode_indices: List[int], 
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Compare policy behavior across multiple episodes
        
        Args:
            episode_indices: List of episode indices to compare
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Policy Strategy Comparison Across Episodes', fontsize=14, fontweight='bold')
        
        # Collect data for all episodes
        episode_summaries = []
        for idx in episode_indices:
            summary = self.tracker.get_hedge_summary(idx)
            if summary:
                summary['episode'] = idx
                episode_summaries.append(summary)
        
        if not episode_summaries:
            print("No episode data available for comparison")
            return None
        
        # 1. Hedge Ratio Comparison
        ax1 = axes[0]
        episodes = [s['episode'] for s in episode_summaries]
        hedge_ratios = [s['hedge_ratio'] for s in episode_summaries]
        total_costs = [s['total_cost'] for s in episode_summaries]
        
        bars = ax1.bar(episodes, hedge_ratios, color=self.colors['hedge'], alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Hedge Ratio')
        ax1.set_title('Hedge Frequency by Episode')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, hedge_ratios):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ratio:.2f}', ha='center', va='bottom')
        
        # 2. Cost vs Hedge Ratio
        ax2 = axes[1]
        scatter = ax2.scatter(hedge_ratios, total_costs, 
                            c=episodes, cmap='viridis', s=100, alpha=0.8, edgecolors='black')
        ax2.set_xlabel('Hedge Ratio')
        ax2.set_ylabel('Total Cost')
        ax2.set_title('Cost vs Hedge Frequency')
        plt.colorbar(scatter, ax=ax2, label='Episode')
        ax2.grid(True, alpha=0.3)
        
        # Add episode labels
        for i, (ratio, cost, ep) in enumerate(zip(hedge_ratios, total_costs, episodes)):
            ax2.annotate(f'Ep{ep}', (ratio, cost), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self, episode_idx: Optional[int] = None) -> str:
        """
        Generate a text summary report of policy behavior
        
        Args:
            episode_idx: Episode index to analyze
            
        Returns:
            Formatted text report
        """
        summary = self.tracker.get_hedge_summary(episode_idx)
        
        if not summary:
            return "No data available for summary report"
        
        report = f"""
Policy Behavior Summary Report
{'='*50}

Episode: {episode_idx if episode_idx is not None else 'Latest'}

Decision Overview:
- Total Decisions: {summary['total_decisions']}
- Hedge Actions: {summary['hedge_actions']}
- Wait Actions: {summary['wait_actions']}
- Hedge Ratio: {summary['hedge_ratio']:.2%}

Performance Metrics:
- Total Cost: {summary['total_cost']:.4f}
- Average Execution Probability: {summary['avg_execution_prob']:.4f}
"""
        
        if 'avg_hedge_cost' in summary:
            report += f"""
Hedge Analysis:
- Average Hedge Cost: {summary['avg_hedge_cost']:.4f}
- Average Price at Hedge: {summary['avg_hedge_price']:.4f}
- Price Range at Hedges: {summary['hedge_price_range'][0]:.4f} - {summary['hedge_price_range'][1]:.4f}
- Hedge Timesteps: {summary['hedge_timesteps']}
"""
        
        return report
    
    def export_visualizations(self, output_dir: str = "visualizations", 
                            episode_idx: Optional[int] = None):
        """
        Export all visualizations to files
        
        Args:
            output_dir: Directory to save visualizations
            episode_idx: Episode index to visualize
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save timeline plot
        fig1 = self.plot_policy_timeline(episode_idx)
        if fig1:
            fig1.savefig(os.path.join(output_dir, f'policy_timeline_ep{episode_idx or "latest"}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig1)
        
        # Generate and save hedge analysis
        fig2 = self.plot_hedge_analysis(episode_idx)
        if fig2:
            fig2.savefig(os.path.join(output_dir, f'hedge_analysis_ep{episode_idx or "latest"}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig2)
        
        # Export summary report
        report = self.generate_summary_report(episode_idx)
        with open(os.path.join(output_dir, f'summary_report_ep{episode_idx or "latest"}.txt'), 'w') as f:
            f.write(report)
        
        print(f"Visualizations exported to {output_dir}/")

# Convenience function to create visualizer from global tracker
def create_policy_visualizer() -> PolicyVisualizer:
    """Create a PolicyVisualizer using the global tracker"""
    from .policy_tracker import global_policy_tracker
    return PolicyVisualizer(global_policy_tracker)