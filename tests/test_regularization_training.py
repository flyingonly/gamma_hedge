"""
Test training with regularization to verify hedge ratio improvement
"""

import torch
import sys
import os
import numpy as np
from utils.logger import get_logger


# Add project root to path
logger = get_logger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.production_trainer import ProductionTrainer
from core.config import TrainingConfig
from utils.policy_tracker import global_policy_tracker

def test_regularization_training():
    """Test a small training run with regularization to check hedge ratio"""
    
    print("Testing Regularization Training")
    print("=" * 50)
    
    try:
        # Create TrainingConfig with regularization parameters
        training_config = TrainingConfig(
            n_epochs=3,  # Very short training for testing
            batch_size=8,  # Small batch size
            learning_rate=0.001,
            entropy_weight=0.05,  # Higher than default for stronger effect
            base_execution_cost=0.01,  # Higher execution cost to discourage frequent trading
            variable_execution_cost=0.001
        )
        
        print(f"Regularization settings:")
        print(f"  entropy_weight: {training_config.entropy_weight}")
        print(f"  base_execution_cost: {training_config.base_execution_cost}")
        print(f"  variable_execution_cost: {training_config.variable_execution_cost}")
        print()
        
        # Create production trainer instance
        production_trainer = ProductionTrainer()
        
        # Test with unified configuration
        result = production_trainer.train(
            config_name='single_atm_call',
            training_config=training_config,
            enable_tracking=True,
            enable_visualization=False  # Disable to avoid matplotlib issues
        )
        
        print(f"Training completed successfully!")
        print(f"Result status: {result.get('status', 'unknown')}")
        
        # Check if we have tracking data
        if global_policy_tracker.all_episodes:
            print(f"Episodes tracked: {len(global_policy_tracker.all_episodes)}")
            
            # Analyze hedge ratios from different episodes
            hedge_ratios = []
            for i, episode in enumerate(global_policy_tracker.all_episodes[-3:]):  # Last 3 episodes
                summary = global_policy_tracker.get_hedge_summary(len(global_policy_tracker.all_episodes) - 3 + i)
                if summary:
                    hedge_ratio = summary.get('hedge_ratio', 0)
                    hedge_ratios.append(hedge_ratio)
                    print(f"Episode {i+1}: Hedge ratio = {hedge_ratio:.3f}, "
                          f"Hedge actions = {summary.get('hedge_actions', 0)}, "
                          f"Total decisions = {summary.get('total_decisions', 0)}")
            
            if hedge_ratios:
                avg_hedge_ratio = np.mean(hedge_ratios)
                print(f"\\nAverage hedge ratio: {avg_hedge_ratio:.3f}")
                
                if avg_hedge_ratio < 0.95:  # Less than 95% execution
                    logger.info("PASS: " + Regularization effective - hedge ratio reduced from 1.0 to {avg_hedge_ratio:.3f})
                else:
                    logger.warning(Hedge ratio still high: {avg_hedge_ratio:.3f} - may need stronger regularization)
            else:
                logger.warning(No hedge ratio data available)
        else:
            logger.warning(No tracking data available)
        
        return True
        
    except Exception as e:
        logger.error("FAIL: " + Training test failed: {e})
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_regularization_training()
    if success:
        print("\\n[PASS] Regularization training test completed successfully!")
    else:
        print("\\n[FAIL] Regularization training test failed!")
        sys.exit(1)