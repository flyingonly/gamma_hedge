#!/usr/bin/env python3
"""
Unified Data Processing Tests
============================

Comprehensive tests for data loading, Greeks preprocessing, and precomputed data flow.
Replaces multiple scattered test files with a single integrated test suite.
"""

import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.options_loader import OptionsDataLoader
from data.greeks_preprocessor import GreeksPreprocessor
from data.precomputed_data_loader import PrecomputedGreeksDataset
from core.config import GreeksConfig


class TestUnifiedDataProcessing(unittest.TestCase):
    """Integrated tests for the unified data processing pipeline"""
    
    def setUp(self):
        """Set up test configuration and instances"""
        from core.config import create_config
        full_config = create_config()
        self.config = full_config.greeks
        self.test_weekly_code = '3CN5'
        self.test_option_key = 'CALL_111.0'
        
    def test_options_data_loader(self):
        """Test options data loader initialization and basic functionality"""
        loader = OptionsDataLoader(
            underlying_data_dir=self.config.underlying_data_dir,
            options_data_dir=self.config.options_data_dir,
            mapping_file=self.config.mapping_file
        )
        
        # Test initialization
        self.assertIsNotNone(loader)
        
        # Test available data discovery
        underlyings = loader.get_available_underlyings()
        self.assertGreater(len(underlyings), 0, "Should find available underlying assets")
        
        # Test underlying data loading
        if len(underlyings) > 0:
            underlying_data = loader.load_underlying_data(underlyings[0])
            self.assertIn('timestamps', underlying_data)
            self.assertIn('prices', underlying_data)
            self.assertGreater(len(underlying_data['timestamps']), 0)
        
    def test_greeks_preprocessing(self):
        """Test Greeks preprocessing functionality"""
        preprocessor = GreeksPreprocessor(self.config)
        
        # Test initialization
        self.assertIsNotNone(preprocessor)
        
        # Test weekly code discovery
        weekly_codes = preprocessor.discover_weekly_codes()
        self.assertGreater(len(weekly_codes), 0, "Should find available weekly codes")
        
        # Test single option preprocessing
        if self.test_weekly_code in weekly_codes:
            options = preprocessor.discover_options_for_weekly(self.test_weekly_code)
            if len(options) > 0:
                option_type, strike = options[0]
                
                # Test preprocessing
                output_file = preprocessor.preprocess_single_option(
                    self.test_weekly_code, option_type, strike
                )
                
                # Verify output file exists and contains data
                self.assertTrue(os.path.exists(output_file))
                
                # Load and verify data content
                data = np.load(output_file)
                required_keys = ['timestamps', 'underlying_prices', 'option_prices', 
                               'delta', 'gamma', 'theta', 'vega', 'implied_volatility']
                
                for key in required_keys:
                    self.assertIn(key, data.keys(), f"Missing required key: {key}")
                    self.assertGreater(len(data[key]), 0, f"Empty data for {key}")
    
    def test_precomputed_greeks_dataset(self):
        """Test precomputed Greeks dataset loading"""
        # First ensure we have preprocessed data
        preprocessor = GreeksPreprocessor(self.config)
        weekly_codes = preprocessor.discover_weekly_codes()
        
        if self.test_weekly_code in weekly_codes:
            # Preprocess at least one option
            options = preprocessor.discover_options_for_weekly(self.test_weekly_code)
            if len(options) > 0:
                option_type, strike = options[0]
                preprocessor.preprocess_single_option(
                    self.test_weekly_code, option_type, strike
                )
                
                # Test precomputed dataset loading
                portfolio_positions = {
                    f"{self.test_weekly_code}/{option_type}_{strike}": 1.0
                }
                
                dataset = PrecomputedGreeksDataset(
                    portfolio_positions=portfolio_positions,
                    sequence_length=30  # Reduce to fit available data
                )
                
                # Verify dataset properties
                self.assertGreater(len(dataset), 0, "Dataset should contain training sequences")
                self.assertEqual(dataset.n_assets, 1)
                self.assertEqual(dataset.sequence_length, 30)
                
                # Test data sample
                if len(dataset) > 0:
                    sample = dataset[0]
                    self.assertIsNotNone(sample.prices)
                    self.assertIsNotNone(sample.holdings)
                    # Verify sequence has valid length (may be shorter due to available data)
                    self.assertGreater(sample.prices.shape[0], 0)
    
    def test_end_to_end_data_flow(self):
        """Test complete data processing flow from raw to training-ready"""
        # Step 1: Load options data
        loader = OptionsDataLoader(
            underlying_data_dir=self.config.underlying_data_dir,
            options_data_dir=self.config.options_data_dir,
            mapping_file=self.config.mapping_file
        )
        
        underlyings = loader.get_available_underlyings()
        self.assertGreater(len(underlyings), 0)
        
        # Step 2: Preprocess Greeks
        preprocessor = GreeksPreprocessor(self.config)
        weekly_codes = preprocessor.discover_weekly_codes()
        
        if self.test_weekly_code in weekly_codes:
            options = preprocessor.discover_options_for_weekly(self.test_weekly_code)
            if len(options) > 0:
                option_type, strike = options[0]
                
                # Preprocess
                output_file = preprocessor.preprocess_single_option(
                    self.test_weekly_code, option_type, strike
                )
                self.assertTrue(os.path.exists(output_file))
                
                # Step 3: Create training dataset
                portfolio_positions = {
                    f"{self.test_weekly_code}/{option_type}_{strike}": 1.0
                }
                
                dataset = PrecomputedGreeksDataset(
                    portfolio_positions=portfolio_positions,
                    sequence_length=30
                )
                
                # Verify end-to-end flow
                self.assertGreater(len(dataset), 0)
                
                if len(dataset) > 0:
                    sample = dataset[0]
                    # Verify sample structure matches training requirements
                    self.assertEqual(len(sample.prices.shape), 2)  # (seq_len, n_assets)
                    self.assertEqual(len(sample.holdings.shape), 2)  # (seq_len, n_assets)
                    self.assertEqual(sample.prices.shape[0], 30)
                    self.assertEqual(sample.holdings.shape[0], 30)


if __name__ == '__main__':
    unittest.main(verbosity=2)