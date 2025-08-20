#!/usr/bin/env python3
"""
Unit tests for policy network functionality
Uses the new interface architecture and test configuration
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from tests.test_config import TEST_CONFIG, create_mock_market_data
from common.interfaces import DataResult

def test_policy_network_creation():
    """Test policy network creation and basic functionality"""
    print("Testing Policy Network Creation")
    print("-" * 40)
    
    try:
        from models.policy_network import PolicyNetwork
        
        # Test parameters from config
        n_assets = TEST_CONFIG.n_assets
        input_dim = n_assets * 3  # prev_holding, current_holding, price
        
        # Create policy network
        network = PolicyNetwork(input_dim=input_dim)
        print(f"✅ Policy network created with input_dim={input_dim}")
        
        # Test forward pass
        batch_size = TEST_CONFIG.batch_size
        prev_holding = torch.randn(batch_size, n_assets)
        current_holding = torch.randn(batch_size, n_assets)
        price = torch.randn(batch_size, n_assets)
        
        with torch.no_grad():
            output = network(prev_holding, current_holding, price)
            print(f"✅ Forward pass successful - output shape: {output.shape}")
            
            # Verify output properties
            assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {output.shape}"
            assert torch.all(output >= 0) and torch.all(output <= 1), "Output should be in [0,1] range"
            print("✅ Output validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_policy_network_integration():
    """Test policy network with mock data"""
    print("\nTesting Policy Network Integration")
    print("-" * 40)
    
    try:
        from models.policy_network import PolicyNetwork
        from common.adapters import to_data_result
        
        # Create mock data
        prices, holdings = create_mock_market_data(
            n_samples=TEST_CONFIG.n_samples,
            n_timesteps=TEST_CONFIG.n_timesteps,
            n_assets=TEST_CONFIG.n_assets
        )
        
        # Convert to DataResult format
        data = DataResult(prices=prices, holdings=holdings)
        print(f"✅ Mock data created: {data.prices.shape}")
        
        # Create network
        input_dim = TEST_CONFIG.n_assets * 3
        network = PolicyNetwork(input_dim=input_dim)
        
        # Test with batch data
        sample_data = data[0]  # Get first sample
        batch_size, n_timesteps, n_assets = sample_data.prices.shape
        
        # Simulate trading loop
        current_holding = torch.zeros(1, n_assets)  # Start with zero holding
        
        for t in range(min(5, n_timesteps)):  # Test first 5 timesteps
            current_price = sample_data.prices[:, t, :].unsqueeze(0)
            required_holding = sample_data.holdings[:, t, :].unsqueeze(0)
            prev_holding = current_holding.clone()
            
            with torch.no_grad():
                exec_prob = network(prev_holding, required_holding, current_price)
                
                # Update holding based on execution probability
                current_holding = exec_prob * required_holding + (1 - exec_prob) * current_holding
                
        print("✅ Trading simulation successful")
        print(f"✅ Final holding: {current_holding[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_policy_network_backwards_compatibility():
    """Test backwards compatibility with legacy interfaces"""
    print("\nTesting Backwards Compatibility")
    print("-" * 40)
    
    try:
        from models.policy_network import PolicyNetwork
        from data.data_types import MarketData
        from common.adapters import to_data_result
        
        # Create network
        input_dim = TEST_CONFIG.n_assets * 3
        network = PolicyNetwork(input_dim=input_dim)
        
        # Test with legacy MarketData format
        prices, holdings = create_mock_market_data(n_samples=1)
        legacy_data = MarketData(prices=prices[0], holdings=holdings[0])
        
        # Convert to new format
        data_result = to_data_result(legacy_data)
        print("✅ Legacy format conversion successful")
        
        # Test network with converted data
        batch_data = data_result.prices[:1]  # Single timestep
        current_holding = torch.zeros(1, TEST_CONFIG.n_assets)
        required_holding = data_result.holdings[:1]
        
        with torch.no_grad():
            output = network(current_holding, required_holding, batch_data)
            print(f"✅ Legacy compatibility test passed - output: {output.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backwards compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all unit tests for policy network"""
    print("🧪 Policy Network Unit Tests")
    print("=" * 50)
    
    tests = [
        test_policy_network_creation,
        test_policy_network_integration,
        test_policy_network_backwards_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                failed += 1
                print("❌ FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED with exception: {e}\n")
    
    # Summary
    total = passed + failed
    print("📊 Test Summary")
    print("-" * 30)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    print(f"\n🎯 Overall Result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)