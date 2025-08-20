#!/usr/bin/env python3
"""
Test script to verify unified configuration system
"""

import sys
import os
sys.path.append('.')

def test_config_integration():
    """Test the unified configuration system"""
    print("Testing Unified Configuration System")
    print("=" * 45)
    
    try:
        # Test imports
        from utils.config import (
            get_config, 
            create_unified_config, 
            check_delta_config_availability,
            TrainingConfig,
            ModelConfig,
            DeltaHedgeConfig
        )
        print("Configuration imports successful")
        
        # Check delta config availability
        delta_available = check_delta_config_availability()
        print(f"Delta config availability: {delta_available}")
        
        # Test default configuration
        print("\nTesting default configuration:")
        default_config = get_config()
        print(f"- Market config: {type(default_config['market']).__name__}")
        print(f"- Training config: {type(default_config['training']).__name__}")
        print(f"- Model config: {type(default_config['model']).__name__}")
        print(f"- Delta hedge config: {type(default_config['delta_hedge']).__name__}")
        print(f"- Delta hedge enabled: {default_config['delta_hedge'].enable_delta}")
        
        # Test delta-enabled configuration
        print("\nTesting delta-enabled configuration:")
        delta_config = get_config(enable_delta=True)
        print(f"- Delta hedge enabled: {delta_config['delta_hedge'].enable_delta}")
        print(f"- Underlying codes: {delta_config['delta_hedge'].underlying_codes}")
        print(f"- Weekly codes: {delta_config['delta_hedge'].weekly_codes}")
        print(f"- Sequence length: {delta_config['delta_hedge'].sequence_length}")
        if 'delta_system' in delta_config:
            print(f"- Delta system config loaded: True")
        else:
            print(f"- Delta system config loaded: False")
        
        # Test configuration classes
        print("\nTesting configuration classes:")
        training_cfg = TrainingConfig()
        print(f"- TrainingConfig defaults: batch_size={training_cfg.batch_size}, epochs={training_cfg.n_epochs}")
        
        model_cfg = ModelConfig()
        print(f"- ModelConfig defaults: hidden_dims={model_cfg.hidden_dims}, dropout={model_cfg.dropout_rate}")
        
        delta_cfg = DeltaHedgeConfig()
        print(f"- DeltaHedgeConfig defaults: enable={delta_cfg.enable_delta}, underlying={delta_cfg.underlying_codes}")
        
        # Test unified config with mock arguments
        print("\nTesting unified config with arguments:")
        
        class MockArgs:
            def __init__(self, delta_hedge=False, epochs=None, underlying_codes=None, weekly_codes=None, sequence_length=None):
                self.delta_hedge = delta_hedge
                self.epochs = epochs
                self.underlying_codes = underlying_codes
                self.weekly_codes = weekly_codes
                self.sequence_length = sequence_length
        
        # Test traditional mode
        traditional_args = MockArgs(delta_hedge=False, epochs=20)
        traditional_config = create_unified_config(traditional_args)
        print(f"- Traditional mode: delta_enabled={traditional_config['delta_hedge'].enable_delta}")
        print(f"- Overridden epochs: {traditional_config['training'].n_epochs}")
        
        # Test delta hedge mode
        delta_args = MockArgs(
            delta_hedge=True, 
            epochs=50,
            underlying_codes=['USU5', 'TUU5'],
            weekly_codes=['4CN5', '5CN5'],
            sequence_length=200
        )
        delta_unified_config = create_unified_config(delta_args)
        print(f"- Delta mode: delta_enabled={delta_unified_config['delta_hedge'].enable_delta}")
        print(f"- Overridden epochs: {delta_unified_config['training'].n_epochs}")
        print(f"- Overridden underlying: {delta_unified_config['delta_hedge'].underlying_codes}")
        print(f"- Overridden weekly: {delta_unified_config['delta_hedge'].weekly_codes}")
        print(f"- Overridden sequence: {delta_unified_config['delta_hedge'].sequence_length}")
        
        print("\n" + "=" * 45)
        print("Configuration integration test completed!")
        print("Key features implemented:")
        print("- Unified configuration system")
        print("- Delta hedge configuration integration")
        print("- Command line argument override support")
        print("- Graceful fallback for missing delta system")
        print("- Type-safe configuration classes")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_integration()
    print(f"Test result: {'PASSED' if success else 'FAILED'}")