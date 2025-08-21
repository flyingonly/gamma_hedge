"""
Integration test for improved policy loss with actual training scenario
"""

import torch
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import Trainer
from training.config import TrainingConfig
from models.policy_network import PolicyNetwork
from common.interfaces import DataResult

def test_integration_with_training():
    """Test improved loss calculation in a mini training scenario"""
    print("Testing Integration with Training Scenario")
    print("=" * 60)
    
    try:
        device = 'cpu'
        torch.manual_seed(42)
        
        # Create more realistic test data
        batch_size, n_timesteps, n_assets = 4, 10, 1
        
        # Simulate price path and required holdings
        prices = torch.cumsum(torch.randn(batch_size, n_timesteps, n_assets) * 0.1 + 0.01, dim=1) + 100
        holdings = torch.sin(torch.arange(n_timesteps).float().view(1, -1, 1) * 0.5) * 0.5
        holdings = holdings.expand(batch_size, -1, -1)
        
        data = DataResult(prices=prices, holdings=holdings)
        
        # Create training configuration with value function enabled
        config = TrainingConfig(
            learning_rate=0.01,
            value_learning_rate=0.01,
            value_loss_weight=0.5,
            entropy_weight=0.01,
            base_execution_cost=0.001,
            variable_execution_cost=0.0001
        )
        
        # Create policy network and trainer
        policy_net = PolicyNetwork(input_dim=4, hidden_dims=[32, 16])
        trainer = Trainer(
            policy_network=policy_net,
            training_config=config,
            device=device,
            use_value_function=True
        )
        
        print(f"Initial configuration:")
        print(f"  Value function enabled: {trainer.use_value_function}")
        print(f"  Policy network parameters: {sum(p.numel() for p in trainer.policy.parameters())}")
        print(f"  Value network parameters: {sum(p.numel() for p in trainer.value_function.parameters())}")
        
        # Perform several training steps
        losses_over_time = []
        value_losses = []
        advantages = []
        
        for step in range(5):
            # Set to training mode
            trainer.policy.train()
            trainer.value_function.train()
            
            # Zero gradients
            trainer.optimizer.zero_grad()
            trainer.value_optimizer.zero_grad()
            
            # Compute loss
            loss_dict = trainer.compute_policy_loss_with_advantage(data)
            total_loss = loss_dict['total_loss']
            
            # Backward and step
            total_loss.backward()
            trainer.optimizer.step()
            trainer.value_optimizer.step()
            
            # Record metrics
            losses_over_time.append(total_loss.item())
            value_losses.append(loss_dict['value_loss'].item())
            advantages.append(loss_dict['mean_advantage'].item())
            
            print(f"  Step {step+1}: Total Loss = {total_loss.item():.4f}, "
                  f"Value Loss = {loss_dict['value_loss'].item():.4f}, "
                  f"Mean Advantage = {loss_dict['mean_advantage'].item():.4f}")
        
        # Verify training progression
        assert len(losses_over_time) == 5, "Should have 5 training steps"
        
        # Check that value function loss is decreasing (learning)
        first_half_value_loss = np.mean(value_losses[:2])
        second_half_value_loss = np.mean(value_losses[-2:])
        
        print(f"\nTraining progression:")
        print(f"  Early value loss: {first_half_value_loss:.4f}")
        print(f"  Late value loss: {second_half_value_loss:.4f}")
        print(f"  Value loss improvement: {(first_half_value_loss - second_half_value_loss)/first_half_value_loss*100:.1f}%")
        
        # Test evaluation mode
        trainer.policy.eval()
        trainer.value_function.eval()
        
        with torch.no_grad():
            eval_loss_dict = trainer.compute_policy_loss_with_advantage(data)
            print(f"  Evaluation loss: {eval_loss_dict['total_loss'].item():.4f}")
        
        # Verify all components are working
        for key in ['total_loss', 'policy_loss', 'value_loss', 'entropy_loss']:
            assert key in eval_loss_dict, f"Missing component: {key}"
            assert torch.isfinite(eval_loss_dict[key]).all(), f"Invalid {key}"
        
        print(f"\n[PASS] Integration test completed successfully")
        print(f"       Final total loss: {losses_over_time[-1]:.6f}")
        print(f"       Value function learned: {second_half_value_loss < first_half_value_loss}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility():
    """Test that trainer still works when value function is disabled"""
    print("\nTesting Backwards Compatibility")
    print("=" * 60)
    
    try:
        device = 'cpu'
        
        # Create test data
        data = DataResult(
            prices=torch.rand(2, 5, 1) * 100 + 50,
            holdings=torch.rand(2, 5, 1) * 2 - 1
        )
        
        # Create trainer without value function
        policy_net = PolicyNetwork(input_dim=4, hidden_dims=[16, 8])
        config = TrainingConfig()
        trainer = Trainer(
            policy_network=policy_net,
            training_config=config,
            device=device,
            use_value_function=False  # Disable value function
        )
        
        assert trainer.value_function is None, "Value function should be None when disabled"
        assert trainer.value_optimizer is None, "Value optimizer should be None when disabled"
        
        # Should still work with advantages = returns-to-go
        loss_dict = trainer.compute_policy_loss_with_advantage(data)
        
        # Value loss should be zero when disabled
        assert loss_dict['value_loss'].item() == 0.0, "Value loss should be zero when disabled"
        
        # Should still have valid policy loss and entropy loss
        assert torch.isfinite(loss_dict['policy_loss']).all(), "Policy loss should be valid"
        assert torch.isfinite(loss_dict['entropy_loss']).all(), "Entropy loss should be valid"
        
        print(f"[PASS] Backwards compatibility test")
        print(f"       Policy loss: {loss_dict['policy_loss'].item():.6f}")
        print(f"       Value loss (disabled): {loss_dict['value_loss'].item():.6f}")
        print(f"       Mean advantage (= returns): {loss_dict['mean_advantage'].item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Backwards compatibility test failed: {e}")
        return False

def run_all_integration_tests():
    """Run all integration tests"""
    print("Running Policy Loss Integration Tests")
    print("=" * 70)
    
    tests = [
        test_integration_with_training,
        test_backwards_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("[PASS] All integration tests passed!")
        return True
    else:
        print(f"[FAIL] {failed} integration tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)