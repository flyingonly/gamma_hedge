# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **gamma hedging and optimal execution research project** that combines:
- Machine learning-based optimal execution strategies using PyTorch
- Black-Scholes option pricing and Greeks calculation
- Weekly options data processing and analysis
- Market simulation and policy evaluation

The codebase implements reinforcement learning approaches to optimize trade execution timing and minimize market impact costs.

## Common Commands

### Training and Execution
```bash
# Train the model from scratch
python main.py

# Resume training from latest checkpoint
python main.py --resume

# Train for specific number of epochs
python main.py --epochs 50

# Evaluate model without training
python main.py --eval-only

# Load and evaluate best model
python main.py --eval-only --load-best

# Resume from specific checkpoint
python main.py --checkpoint checkpoints/checkpoint_epoch_20_*.pth
```

### Data Processing
```bash
# Process weekly options data
python csv_process/weekly_options_processor.py

# Run separated NPZ data processing
python csv_process/run_separated_npz.py

# Test Black-Scholes calculations
python tools/test.py
```

### Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

## Code Architecture

### Core Components

**Main Training Pipeline** (`main.py`):
- Orchestrates training, evaluation, and visualization
- Handles command-line arguments and checkpoint management
- Integrates all components for end-to-end execution

**Policy Network** (`models/policy_network.py`):
- Neural network implementing execution policy u*(θ; W(t_{i-1}), W, S, H)
- Takes previous holdings, current holdings, and prices as input
- Outputs execution probability (0 to 1) using sigmoid activation

**Market Simulation** (`data/market_simulator.py`):
- Geometric Brownian Motion price path generation
- Configurable market parameters (volatility, drift, assets)
- Generates realistic holding schedules with noise

**Training Framework** (`training/trainer.py`):
- REINFORCE-style policy gradient implementation
- Automatic checkpoint saving and resuming
- Validation tracking and best model selection

**Options Toolkit** (`tools/black_scholes.py`):
- Complete Black-Scholes implementation with Greeks
- Input validation and utility functions
- Designed for options analysis and risk management

**Data Processing** (`csv_process/`):
- Processes weekly options Excel files into NPZ format
- Handles monthly data mapping and futures symbols
- Chinese comments indicate options trading focus

### Key Design Patterns

**Configuration System** (`utils/config.py`):
- Centralized configuration using dataclasses
- Separate configs for market simulation, training, and model
- Default values optimized for research use

**Modular Data Loading** (`data/data_loader.py`):
- Batch generation for training
- Integration with PyTorch DataLoader
- Configurable market simulation parameters

**Checkpoint Management**:
- Automatic saving every N epochs (configurable)
- Best model tracking based on validation loss
- Resume capability with `--resume` flag
- Latest, best, and final checkpoints maintained

### Input/Output Structure

**Model Input Dimensions**:
- Input: `n_assets * 3` (prev_holding, current_holding, price)
- Hidden layers: configurable (default: [128, 64, 32])
- Output: single execution probability

**Data Format**:
- Prices: `(batch_size, n_timesteps, n_assets)`
- Holdings: `(batch_size, n_timesteps, n_assets)`  
- NPZ files store processed options data

**Checkpoint Structure**:
- Model state dict, optimizer state, training history
- Epoch information for resuming
- Device information for proper loading

## Important Notes

- The project uses PyTorch for neural networks and training
- Market simulation uses GBM for realistic price dynamics
- Options data processing handles real-world weekly options contracts
- Training uses policy gradients with immediate cost as reward signal
- Checkpoints are saved to `checkpoints/` directory automatically
- All random seeds are set for reproducibility (seed=42)

## Greeks Optimization System

The project includes a unified DeltaHedgeDataset with optimized Greeks calculation that pre-computes volatility, delta, and other Greeks for individual options, significantly improving dataset loading performance while maintaining backward compatibility.

### Key Components

**Greeks Preprocessor** (`data/greeks_preprocessor.py`):
- Pre-calculates Greeks (Delta, Gamma, Theta, Vega) for individual options
- Saves preprocessed data to NPZ files for fast loading
- Automatic cache management and incremental updates

**Unified DeltaHedgeDataset** (`data/data_loader.py`):
- Single dataset class supporting both legacy and optimized loading
- Automatically detects position format and uses appropriate strategy
- Loads preprocessed Greeks files for specific options
- Converts legacy weekly positions to specific options for consistency
- Supports multiple underlying assets in a single portfolio
- Dynamically calculates portfolio combinations during dataset creation

**Batch Preprocessing** (`scripts/preprocess_greeks.py`):
- Command-line tool for bulk Greeks preprocessing
- Discovers all available options automatically
- Supports targeted preprocessing of specific weekly codes

### File Organization

```
data/
├── preprocessed_greeks/
│   ├── 3CN5/
│   │   ├── CALL_111.0_greeks.npz     # Individual option Greeks
│   │   ├── PUT_111.5_greeks.npz
│   │   └── ...
│   ├── 3IN5/
│   │   └── ...
│   └── cache_metadata.json           # Preprocessing metadata
```

### Usage

**Unified Dataset Creation (Multiple Formats)**:
```python
from data.data_loader import create_delta_data_loader

# Format 1: Specific options (recommended for precision)
option_positions = {
    '3CN5/CALL_111.0': 1.0,    # Long 1 call at strike 111.0
    '3CN5/PUT_111.5': -0.5,    # Short 0.5 put at strike 111.5
    '3IN5/CALL_106.5': 2.0,    # Long 2 calls from different underlying
}

# Format 2: Legacy weekly positions (auto-converted)
weekly_positions = {'3CN5': 1.0, '3IN5': -0.5}

# Format 3: Legacy list format (auto-converted)
weekly_list = ['3CN5', '3IN5', '3MN5']

# Create data loader (supports all formats)
data_loader = create_delta_data_loader(
    batch_size=32,
    positions=option_positions,  # or weekly_positions or weekly_list
    start_date='2025-01-01',
    use_preprocessed_greeks=True,  # Use optimized loading
    auto_preprocess=True           # Auto-generate missing Greeks files
)
```

**Manual Preprocessing**:
```bash
# Preprocess all available options
python scripts/preprocess_greeks.py

# Preprocess specific weekly codes
python scripts/preprocess_greeks.py --codes 3CN5 3IN5
```

### Performance Benefits

- **Faster Loading**: Avoids recalculating Black-Scholes formulas during training
- **Cache Reuse**: Same option Greeks can be reused across different portfolios
- **Incremental Updates**: Only reprocesses files when underlying data changes
- **Memory Efficiency**: Only loads required date ranges and options
- **Backward Compatibility**: Legacy weekly position formats automatically converted
- **Multi-Underlying Support**: Single portfolio can include options from different underlying assets

### Multi-Underlying Asset Support

The system supports portfolios with options from different underlying assets:

```python
# Portfolio spanning multiple underlying assets
portfolio = {
    '3CN5/CALL_111.0': 1.0,
    '3IN5/PUT_107.0': -0.5,
    '3MN5/CALL_110.0': 2.0,
    '3WN5/PUT_103.5': -1.0,
}

# Each underlying asset gets its own column in the price/holdings tensors
# Portfolio delta is calculated by aggregating across all options
```