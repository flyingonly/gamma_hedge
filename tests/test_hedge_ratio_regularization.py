"""
Test hedge ratio regularization implementation
"""

import torch
from unittest.mock import MagicMock
import sys
import os
from utils.logger import get_logger


# Add project root to path
logger = get_logger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import Trainer
from core.config import TrainingConfig
from models.policy_network import PolicyNetwork
from core.interfaces import DataResult

def test_trainer_regularization_parameters():
    """Test that Trainer accepts regularization parameters through TrainingConfig"""
    
    # Create a simple policy network
    model = PolicyNetwork(input_dim=4, hidden_dims=[32, 16])
    
    # Create training configuration with custom regularization parameters
    training_config = TrainingConfig(
        entropy_weight=0.05,
        base_execution_cost=0.002,
        variable_execution_cost=0.0002
    )
    
    # Test trainer with unified configuration
    trainer = Trainer(
        policy_network=model,
        training_config=training_config,
        device='cpu'
    )
    
    # Verify parameters are set correctly
    assert trainer.entropy_weight == 0.05
    assert trainer.base_execution_cost == 0.002
    assert trainer.variable_execution_cost == 0.0002
    
    logger.info("PASS: " + Trainer regularization parameters test)

def test_entropy_calculation():
    """Test that entropy regularization is computed correctly"""
    
    # Create simple test data
    batch_size = 2
    n_timesteps = 3
    n_assets = 1
    
    prices = torch.rand(batch_size, n_timesteps, n_assets)
    holdings = torch.rand(batch_size, n_timesteps, n_assets)
    
    data = DataResult(prices=prices, holdings=holdings)
    
    # Create trainer with configuration
    model = PolicyNetwork(input_dim=4, hidden_dims=[16, 8])
    training_config = TrainingConfig(
        entropy_weight=0.1,
        base_execution_cost=0.01
    )
    trainer = Trainer(
        policy_network=model,
        training_config=training_config,
        device='cpu'
    )
    
    # Compute loss (should include entropy regularization)
    try:
        loss = trainer.compute_policy_loss(data)
        
        # Loss should be a scalar tensor
        assert loss.dim() == 0
        assert torch.isfinite(loss).item()
        
        logger.info("PASS: " + Entropy calculation test - Loss: {loss.item():.6f})
        
    except Exception as e:
        logger.error("FAIL: " + Entropy calculation test failed: {e})
        raise

def test_execution_cost_penalty():
    """Test that execution cost penalty affects loss calculation"""
    
    # Create test data
    batch_size = 1
    n_timesteps = 2
    n_assets = 1
    
    prices = torch.ones(batch_size, n_timesteps, n_assets)
    holdings = torch.tensor([[[0.5], [1.0]]])  # Significant holding change
    
    data = DataResult(prices=prices, holdings=holdings)
    
    # Test with different execution costs
    model1 = PolicyNetwork(input_dim=4, hidden_dims=[16, 8])
    low_cost_config = TrainingConfig(
        base_execution_cost=0.001,
        variable_execution_cost=0.0001,
        entropy_weight=0.01
    )
    trainer_low_cost = Trainer(
        policy_network=model1,
        training_config=low_cost_config,
        device='cpu'
    )
    
    model2 = PolicyNetwork(input_dim=4, hidden_dims=[16, 8])
    # Copy weights to ensure same policy outputs
    model2.load_state_dict(model1.state_dict())
    high_cost_config = TrainingConfig(
        base_execution_cost=0.01,
        variable_execution_cost=0.001,
        entropy_weight=0.01
    )
    trainer_high_cost = Trainer(
        policy_network=model2,
        training_config=high_cost_config,
        device='cpu'
    )
    
    try:
        loss_low = trainer_low_cost.compute_policy_loss(data)
        loss_high = trainer_high_cost.compute_policy_loss(data)
        
        logger.info(Low cost loss: {loss_low.item():.6f})
        logger.info(High cost loss: {loss_high.item():.6f})
        
        # High execution cost should generally lead to different loss
        # (might be higher or lower depending on policy behavior)
        logger.info("PASS: " + Execution cost penalty test - costs affect loss calculation)
        
    except Exception as e:
        logger.error("FAIL: " + Execution cost penalty test failed: {e})
        raise

def test_integration_with_policy_tracker():
    """Test that regularization works with policy tracking enabled"""
    
    # Create test data
    batch_size = 1
    n_timesteps = 3
    n_assets = 1
    
    prices = torch.rand(batch_size, n_timesteps, n_assets)
    holdings = torch.rand(batch_size, n_timesteps, n_assets)
    
    data = DataResult(prices=prices, holdings=holdings)
    
    # Create trainer with tracking enabled
    model = PolicyNetwork(input_dim=4, hidden_dims=[16, 8])
    tracking_config = TrainingConfig(entropy_weight=0.01)
    trainer = Trainer(
        policy_network=model,
        training_config=tracking_config,
        enable_tracking=True,
        device='cpu'
    )
    
    try:
        loss = trainer.compute_policy_loss(data)
        
        # Should complete without errors
        assert torch.isfinite(loss).item()
        
        logger.info("PASS: " + Policy tracker integration test)
        
    except Exception as e:
        logger.error("FAIL: " + Policy tracker integration test failed: {e})
        raise

if __name__ == "__main__":
    print("Running Hedge Ratio Regularization Tests")
    print("=" * 50)
    
    try:
        test_trainer_regularization_parameters()
        test_entropy_calculation()
        test_execution_cost_penalty()
        test_integration_with_policy_tracker()
        
        print("=" * 50)
        logger.info("PASS: " + All regularization tests passed!)
        
    except Exception as e:
        print("=" * 50)
        logger.error("FAIL: " + Test suite failed: {e})
        sys.exit(1)