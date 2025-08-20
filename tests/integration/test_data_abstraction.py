#!/usr/bin/env python3
"""
Comprehensive integration tests for data abstraction layer
Tests the new DataResult interface, adapters, and dataset integration
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tests.test_config import TEST_CONFIG, create_mock_market_data, create_mock_option_positions

def test_data_result_interface():
    """Test DataResult interface functionality"""
    print("Testing DataResult Interface")
    print("-" * 40)
    
    try:
        from common.interfaces import DataResult
        
        # Create mock data
        prices, holdings = create_mock_market_data()
        
        # Test DataResult creation
        data = DataResult(prices=prices, holdings=holdings, 
                         metadata={'test': True, 'version': '1.0'})
        
        print(f"✅ DataResult created: {data.prices.shape}")
        
        # Test properties
        assert data.n_samples == prices.shape[0], "n_samples mismatch"
        assert data.sequence_length == prices.shape[1], "sequence_length mismatch"  
        assert data.n_assets == prices.shape[2], "n_assets mismatch"
        print("✅ Properties validation passed")
        
        # Test indexing
        sample = data[0]
        assert isinstance(sample, DataResult), "Indexing should return DataResult"
        assert sample.prices.shape == (1, prices.shape[1], prices.shape[2]), "Single sample shape incorrect"
        print("✅ Indexing functionality passed")
        
        # Test device movement
        if torch.cuda.is_available():
            data_gpu = data.to('cuda')
            assert data_gpu.prices.device.type == 'cuda', "GPU device movement failed"
            data_cpu = data_gpu.to('cpu')
            assert data_cpu.prices.device.type == 'cpu', "CPU device movement failed"
            print("✅ Device movement passed")
        else:
            print("✅ Device movement skipped (CUDA not available)")
        
        # Test shape validation
        try:
            DataResult(prices=prices, holdings=holdings[:, :, :2])  # Wrong shape
            assert False, "Should have raised validation error"
        except ValueError:
            print("✅ Shape validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ DataResult test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adapters():
    """Test data format adapters"""
    print("\nTesting Data Format Adapters")
    print("-" * 40)
    
    try:
        from common.adapters import to_data_result, LegacyDataAdapter
        from data.data_types import MarketData
        from common.interfaces import DataResult
        
        # Create test data in different formats
        prices, holdings = create_mock_market_data(n_samples=1)
        
        # Test tuple format conversion
        tuple_data = (prices[0], holdings[0])
        result1 = to_data_result(tuple_data)
        assert isinstance(result1, DataResult), "Tuple conversion failed"
        print("✅ Tuple format conversion passed")
        
        # Test MarketData format conversion  
        market_data = MarketData(prices=prices[0], holdings=holdings[0])
        result2 = to_data_result(market_data)
        assert isinstance(result2, DataResult), "MarketData conversion failed"
        print("✅ MarketData format conversion passed")
        
        # Test DataResult passthrough
        data_result = DataResult(prices=prices[0], holdings=holdings[0])
        result3 = to_data_result(data_result)
        assert result3 is data_result, "DataResult passthrough failed"
        print("✅ DataResult passthrough passed")
        
        # Test legacy adapter
        adapter = LegacyDataAdapter()
        legacy_tuple = adapter.to_tuple(data_result)
        assert isinstance(legacy_tuple, tuple), "Legacy tuple conversion failed"
        assert len(legacy_tuple) == 2, "Legacy tuple should have 2 elements"
        print("✅ Legacy adapter passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Adapters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_integration():
    """Test dataset integration with new interfaces"""
    print("\nTesting Dataset Integration")
    print("-" * 40)
    
    try:
        from data.data_loader import TradingDataset, create_data_loader
        from data.market_simulator import MarketSimulator
        from data.config import MarketConfig
        from torch.utils.data import DataLoader
        
        # Create market config
        config = MarketConfig(
            n_assets=TEST_CONFIG.n_assets,
            n_timesteps=TEST_CONFIG.n_timesteps,
            initial_price=100.0,
            volatility=0.2
        )
        
        # Test market simulator
        simulator = MarketSimulator(config)
        prices, holdings = simulator.generate_batch(5)
        print(f"✅ Market simulator generated data: {prices.shape}")
        
        # Test TradingDataset
        dataset = TradingDataset(simulator, n_samples=10)
        assert len(dataset) == 10, "Dataset length incorrect"
        
        # Test dataset indexing
        sample = dataset[0]
        from common.interfaces import DataResult
        assert isinstance(sample, DataResult), "Dataset should return DataResult"
        assert 'dataset_type' in sample.metadata, "Metadata missing"
        print("✅ TradingDataset integration passed")
        
        # Test data loader creation
        data_loader = create_data_loader(
            config=config, 
            n_samples=TEST_CONFIG.n_samples, 
            batch_size=TEST_CONFIG.batch_size
        )
        assert isinstance(data_loader, DataLoader), "Should return PyTorch DataLoader"
        
        # Test batch loading
        for batch in data_loader:
            assert isinstance(batch, DataResult), "Batch should be DataResult"
            assert batch.prices.shape[0] <= TEST_CONFIG.batch_size, "Batch size exceeded"
            break  # Test just first batch
        
        print("✅ Data loader integration passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_delta_hedge_integration():
    """Test delta hedge dataset integration if available"""
    print("\nTesting Delta Hedge Integration")  
    print("-" * 40)
    
    try:
        from data.data_loader import check_delta_availability, DeltaHedgeDataset
        
        if not check_delta_availability():
            print("⚠️  Delta hedge not available, skipping tests")
            return True
        
        # Create mock option positions
        option_positions = create_mock_option_positions()
        
        # Test DeltaHedgeDataset creation
        dataset = DeltaHedgeDataset(
            option_positions=option_positions,
            sequence_length=TEST_CONFIG.n_timesteps
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            from common.interfaces import DataResult
            assert isinstance(sample, DataResult), "DeltaHedgeDataset should return DataResult"
            assert sample.metadata['dataset_type'] == 'delta_hedge', "Metadata incorrect"
            print("✅ Delta hedge dataset integration passed")
        else:
            print("⚠️  Delta hedge dataset empty, skipping sample test")
        
        return True
        
    except Exception as e:
        print(f"❌ Delta hedge integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_integration():
    """Test integration with training components"""
    print("\nTesting Training Integration")
    print("-" * 40)
    
    try:
        from training.trainer import Trainer
        from models.policy_network import PolicyNetwork
        from data.data_loader import create_data_loader
        from data.config import MarketConfig
        
        # Create small test configuration
        config = MarketConfig(
            n_assets=2,  # Smaller for faster testing
            n_timesteps=10,
            volatility=0.1
        )
        
        # Create components
        input_dim = config.n_assets * 3
        policy = PolicyNetwork(input_dim=input_dim)
        trainer = Trainer(policy, learning_rate=0.01, device='cpu')
        
        # Create small dataset
        data_loader = create_data_loader(config, n_samples=5, batch_size=2)
        
        # Test single training step
        trainer.policy.train()
        for batch in data_loader:
            loss = trainer.compute_policy_loss(batch)
            assert isinstance(loss, torch.Tensor), "Loss should be tensor"
            assert loss.requires_grad, "Loss should require gradients"
            break
        
        print("✅ Training integration passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Training integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all data abstraction integration tests"""
    print("🧪 Data Abstraction Integration Tests")
    print("=" * 50)
    
    tests = [
        test_data_result_interface,
        test_adapters,
        test_dataset_integration,
        test_delta_hedge_integration,
        test_training_integration,
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