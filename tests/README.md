# Gamma Hedge Test Suite

This directory contains the refactored and organized test suite for the gamma hedge project.

## Test Structure

```
tests/
├── __init__.py              # Test configuration and global setup
├── test_config.py           # Test utilities and mock data generators
├── README.md                # This file
├── unit/                    # Unit tests for individual components
│   └── test_policy_network.py
├── integration/             # Integration tests for component interactions
│   ├── test_data_abstraction.py
│   ├── test_data_loader_integration.py
│   ├── test_policy_network_integration.py
│   ├── test_config_integration.py
│   └── test_system_integration.py
├── functional/              # Functional tests for specific features
│   ├── test_dynamic_delta_dataset.py
│   ├── test_greeks_preprocessing.py
│   ├── test_multi_underlying_support.py
│   ├── test_optimized_real_data.py
│   └── test_real_data_loading.py
└── delta_hedge/             # Delta hedge specific tests
    └── test_delta_hedge_flow.py
```

## Running Tests

### Windows Batch Script (Recommended)
```cmd
# Run all tests with conda environment
test_runner.bat

# Run specific test types
test_runner.bat --type unit
test_runner.bat --type integration
test_runner.bat --verbose
```

### Individual Test Files
```bash
# Run specific test files
python tests/unit/test_policy_network.py
python tests/integration/test_data_abstraction.py
```

## Key Features

- **Unified Interface**: All tests use the new DataResult interface
- **Mock Data**: Centralized mock data generation with TEST_CONFIG
- **Error Handling**: Comprehensive error reporting and debugging
- **Modular Structure**: Clear separation of unit, integration, and functional tests
- **Environment Detection**: Automatic delta hedge availability checking
- **Backwards Compatibility**: Support for legacy data formats through adapters

## Test Configuration

Key parameters in `tests/test_config.py`:
- Device: CPU (consistent testing)
- Batch size: 4 (fast execution)
- Assets: 3, Timesteps: 20
- Timeout: 60 seconds per test

## Migration Notes

This replaces the old test structure with:
- Removed 10+ debug/temporary files
- Consolidated from 19 to 12 core test files
- Standardized interfaces and error handling
- Added proper documentation and examples