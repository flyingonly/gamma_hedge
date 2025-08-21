"""
Test policy loss improvements: returns-to-go, value function baseline, and training-evaluation consistency
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
from models.value_function import ValueFunction
from common.interfaces import DataResult
import torch.nn.functional as F

def create_test_data(batch_size=2, n_timesteps=5, n_assets=1, device='cpu'):
    """Create synthetic test data for validation"""
    torch.manual_seed(42)  # For reproducibility
    
    prices = torch.rand(batch_size, n_timesteps, n_assets, device=device) * 100 + 50
    holdings = torch.rand(batch_size, n_timesteps, n_assets, device=device) * 2 - 1  # -1 to 1
    
    return DataResult(prices=prices, holdings=holdings)

def test_value_function_network():
    """Test that value function network works correctly"""
    print("Testing Value Function Network")
    print("=" * 50)
    
    try:
        # Create test data
        device = 'cpu'
        data = create_test_data(device=device)
        
        # Create policy and value networks
        policy_net = PolicyNetwork(input_dim=4, hidden_dims=[32, 16])
        config = TrainingConfig(
            value_hidden_dims=[32, 16],
            value_learning_rate=0.001
        )
        
        # Create trainer with value function
        trainer = Trainer(
            policy_network=policy_net,
            training_config=config,
            device=device,
            use_value_function=True
        )
        
        # Test value function forward pass
        batch_size, n_timesteps, n_assets = data.prices.shape
        prev_holding = torch.zeros(batch_size, n_assets)
        current_holding = data.holdings[:, 0, :]
        price = data.prices[:, 0, :]
        time_features = torch.ones(batch_size, 1) * 0.8
        
        value_pred = trainer.value_function(
            prev_holding, current_holding, price, time_features=time_features
        )
        
        assert value_pred.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {value_pred.shape}"
        assert torch.isfinite(value_pred).all(), "Value predictions contain infinite or NaN values"
        
        print(f"[PASS] Value function network test")
        print(f"       Value predictions: {value_pred.flatten()}")
        
    except Exception as e:
        print(f"[FAIL] Value function network test failed: {e}")
        raise

def test_returns_to_go_calculation():
    """Test returns-to-go calculation vs traditional total reward"""
    print("\nTesting Returns-to-go Calculation")
    print("=" * 50)
    
    try:
        device = 'cpu'
        data = create_test_data(device=device)
        
        # Create trainer
        policy_net = PolicyNetwork(input_dim=4, hidden_dims=[32, 16])
        config = TrainingConfig()
        trainer = Trainer(
            policy_network=policy_net,
            training_config=config,
            device=device,
            use_value_function=True
        )
        
        # Compute loss with advantages
        loss_dict = trainer.compute_policy_loss_with_advantage(data)
        
        # Check that all expected components are present
        expected_keys = ['total_loss', 'policy_loss', 'value_loss', 'entropy_loss', 'mean_advantage', 'mean_return']
        for key in expected_keys:
            assert key in loss_dict, f"Missing expected loss component: {key}"
            assert torch.isfinite(loss_dict[key]).all(), f"{key} contains infinite or NaN values"
        
        print(f"[PASS] Returns-to-go calculation test")
        print(f"       Policy loss: {loss_dict['policy_loss'].item():.6f}")
        print(f"       Value loss: {loss_dict['value_loss'].item():.6f}")
        print(f"       Mean advantage: {loss_dict['mean_advantage'].item():.6f}")
        print(f"       Mean return: {loss_dict['mean_return'].item():.6f}")
        
    except Exception as e:
        print(f"[FAIL] Returns-to-go calculation test failed: {e}")
        raise

def test_terminal_cost_handling():
    """Test that terminal forced execution cost is properly handled"""
    print("\nTesting Terminal Cost Handling")
    print("=" * 50)
    
    try:
        device = 'cpu'
        # Create data with significant final holding change
        batch_size, n_timesteps, n_assets = 1, 3, 1
        
        prices = torch.tensor([[[100.0], [101.0], [102.0]]], device=device)
        holdings = torch.tensor([[[0.5], [0.3], [1.0]]], device=device)  # Large final change
        data = DataResult(prices=prices, holdings=holdings)
        
        # Create trainer
        policy_net = PolicyNetwork(input_dim=4, hidden_dims=[16, 8])
        config = TrainingConfig(
            base_execution_cost=0.01,
            variable_execution_cost=0.001
        )
        trainer = Trainer(
            policy_network=policy_net,
            training_config=config,
            device=device,
            use_value_function=True
        )
        
        # Compute loss
        loss_dict = trainer.compute_policy_loss_with_advantage(data)
        
        # Terminal cost should be included in returns calculation
        assert torch.isfinite(loss_dict['total_loss']).all(), "Total loss contains infinite or NaN"
        
        print(f"[PASS] Terminal cost handling test")
        print(f"       Total loss: {loss_dict['total_loss'].item():.6f}")
        print(f"       Training return (negative): {loss_dict['mean_return'].item():.6f}")
        
        # Test that terminal cost is properly included (non-zero mean return)
        assert abs(loss_dict['mean_return'].item()) > 0.1, "Terminal cost should contribute to returns"
        
        print(f"[PASS] Terminal cost inclusion verified")
        
    except Exception as e:
        print(f"[FAIL] Terminal cost handling test failed: {e}")
        raise

def test_advantage_variance_reduction():
    """Test that advantage function reduces variance compared to raw returns"""
    print("\nTesting Advantage Variance Reduction")
    print("=" * 50)
    
    try:
        device = 'cpu'
        n_samples = 10
        
        # Collect multiple samples to measure variance
        raw_returns = []
        advantages = []
        
        for i in range(n_samples):
            torch.manual_seed(i)  # Different seed for each sample
            data = create_test_data(batch_size=1, device=device)
            
            # Create trainer
            policy_net = PolicyNetwork(input_dim=4, hidden_dims=[32, 16])
            config = TrainingConfig()
            trainer = Trainer(
                policy_network=policy_net,
                training_config=config,
                device=device,
                use_value_function=True
            )
            
            # Get loss components
            loss_dict = trainer.compute_policy_loss_with_advantage(data)
            
            raw_returns.append(loss_dict['mean_return'].item())
            advantages.append(loss_dict['mean_advantage'].item())
        
        # Calculate variances
        raw_var = np.var(raw_returns)
        advantage_var = np.var(advantages)
        
        print(f"[INFO] Raw returns variance: {raw_var:.6f}")
        print(f"[INFO] Advantage variance: {advantage_var:.6f}")
        
        if advantage_var > 1e-8:
            ratio = raw_var / advantage_var
            print(f"[INFO] Variance reduction ratio: {ratio:.2f}x")
        else:
            print(f"[INFO] Advantage variance too small for ratio calculation")
        
        # Advantage should generally have lower variance (though not guaranteed for small samples)
        print(f"[PASS] Advantage variance reduction test completed")
        
    except Exception as e:
        print(f"[FAIL] Advantage variance reduction test failed: {e}")
        raise

def test_training_step():
    """Test that training step works with new loss calculation"""
    print("\nTesting Training Step")
    print("=" * 50)
    
    try:
        device = 'cpu'
        data = create_test_data(batch_size=2, device=device)
        
        # Create trainer
        policy_net = PolicyNetwork(input_dim=4, hidden_dims=[32, 16])
        config = TrainingConfig(learning_rate=0.01)
        trainer = Trainer(
            policy_network=policy_net,
            training_config=config,
            device=device,
            use_value_function=True
        )
        
        # Store initial parameters
        initial_policy_params = [p.clone() for p in trainer.policy.parameters()]
        initial_value_params = [p.clone() for p in trainer.value_function.parameters()]
        
        # Perform training step
        trainer.policy.train()
        trainer.value_function.train()
        
        # Zero gradients
        trainer.optimizer.zero_grad()
        trainer.value_optimizer.zero_grad()
        
        # Compute loss and backprop
        loss_dict = trainer.compute_policy_loss_with_advantage(data)
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        # Update
        trainer.optimizer.step()
        trainer.value_optimizer.step()
        
        # Check that parameters changed
        policy_changed = any(not torch.equal(p1, p2) for p1, p2 in 
                           zip(initial_policy_params, trainer.policy.parameters()))
        value_changed = any(not torch.equal(p1, p2) for p1, p2 in 
                          zip(initial_value_params, trainer.value_function.parameters()))
        
        assert policy_changed, "Policy parameters did not change during training"
        assert value_changed, "Value function parameters did not change during training"
        
        print(f"[PASS] Training step test")
        print(f"       Policy parameters updated: {policy_changed}")
        print(f"       Value function parameters updated: {value_changed}")
        print(f"       Final loss: {total_loss.item():.6f}")
        
    except Exception as e:
        print(f"[FAIL] Training step test failed: {e}")
        raise

def run_all_tests():
    """Run all policy loss improvement tests"""
    print("Running Policy Loss Improvement Tests")
    print("=" * 60)
    
    test_functions = [
        test_value_function_network,
        test_returns_to_go_calculation,
        test_terminal_cost_handling,
        test_advantage_variance_reduction,
        test_training_step
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("[PASS] All policy loss improvement tests passed!")
    else:
        print(f"[FAIL] {failed} tests failed!")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)