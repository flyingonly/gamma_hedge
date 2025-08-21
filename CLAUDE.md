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
# Production training pipeline (recommended - auto-enables dense mode)
python scripts/run_production_training.py --portfolio single_atm_call --epochs 50

# Dense data mode training (explicit - 442x more training data)
python scripts/run_production_training.py --portfolio single_atm_call --epochs 50 --underlying-dense-mode --sequence-length 1000

# Time-series aligned training with proper train/val/test split
python scripts/run_production_training.py --portfolio single_atm_call --epochs 50 --align-to-daily --split-ratios 0.7 0.2 0.1

# Advanced training with dense mode and daily alignment
python scripts/run_production_training.py --portfolio single_atm_call --epochs 50 --underlying-dense-mode --align-to-daily --min-daily-sequences 100

# List available portfolios
python scripts/run_production_training.py --list-portfolios

# Get portfolio information
python scripts/run_production_training.py --portfolio-info single_atm_call

# Train the model from scratch (legacy)
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

### Testing
```bash
# Run all tests
python scripts/run_tests.py --type all

# Run specific test categories
python scripts/run_tests.py --type unit
python scripts/run_tests.py --type integration
python scripts/run_tests.py --type functional

# Verbose test output
python scripts/run_tests.py --type all --verbose
```

### Data Processing
```bash
# Preprocess Greeks for performance optimization (default: force reprocess all)
python scripts/preprocess_greeks.py

# Preprocess specific weekly codes
python scripts/preprocess_greeks.py --codes 3CN5 3IN5

# Resume from existing cache (skip already processed files)
python scripts/preprocess_greeks.py --codes 3CN5 --resume

# Process all available weekly codes
python scripts/preprocess_greeks.py --all

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

**Configuration System** (`common/greeks_config.py`):
- Unified configuration system using dataclasses
- Simplified from multiple configuration files to single centralized config
- Optimized parameters for Greeks preprocessing and Black-Scholes calculations
- Default values based on research use and market data characteristics

**Unified Options Data Loading** (`data/options_loader.py`):
- Centralized data loading for both underlying and options data
- Automatic discovery of available weekly codes and options
- Integrated time series matching with configurable tolerances
- Streamlined interface replacing multiple data loading pathways

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

## Unified Precomputed Greeks Training System

The project uses a unified training pipeline that directly leverages precomputed Greeks data, eliminating separate integration and preprocessing paths. This ensures maximum accuracy and performance by using real market data throughout the training process.

### Key Components

**Greeks Preprocessor** (`data/greeks_preprocessor.py`):
- Pre-calculates Greeks (Delta, Gamma, Theta, Vega) for individual options
- Saves preprocessed data to NPZ files for fast loading
- Uses historical volatility calculation with configurable fallbacks

**Precomputed Greeks Dataset** (`data/precomputed_data_loader.py`):
- **NEW**: Direct loading of precomputed Greeks data for training
- Converts NPZ files to training sequences with delta-neutral hedging weights
- Supports multi-option portfolios with automatic delta aggregation
- Generates time-aware sequences with proper timestamp alignment

**Simplified Data Loading** (`data/data_loader.py`):
- **SIMPLIFIED**: Removed DeltaHedgeDataset wrapper, directly uses PrecomputedGreeksDataset
- `create_delta_data_loader()` directly creates training-ready data loaders
- Eliminates unnecessary abstraction layers
- Streamlined interface with fewer components

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

**Dual-Mode Dataset Creation**:
```python
from data.data_loader import create_delta_data_loader, create_underlying_dense_data_loader

# Option positions format
option_positions = {
    '3CN5/CALL_111.0': 1.0,    # Long 1 call at strike 111.0
    '3CN5/PUT_111.5': -0.5,    # Short 0.5 put at strike 111.5
    '3IN5/CALL_106.5': 2.0,    # Long 2 calls from different underlying
}

# Method 1: Sparse mode (traditional - 46 data points)
sparse_loader = create_delta_data_loader(
    batch_size=32,
    option_positions=option_positions,
    sequence_length=100,
    underlying_dense_mode=False
)

# Method 2: Dense mode (recommended - 20,335+ data points, 442x increase)
dense_loader = create_delta_data_loader(
    batch_size=32,
    option_positions=option_positions,
    sequence_length=1000,
    underlying_dense_mode=True
)

# Method 3: Dense mode convenience function
dense_loader = create_underlying_dense_data_loader(
    batch_size=32,
    option_positions=option_positions,
    sequence_length=1000
)
```

**Manual Preprocessing**:
```bash
# Preprocess all available options (force reprocess - default)
python scripts/preprocess_greeks.py

# Preprocess specific weekly codes (force reprocess)
python scripts/preprocess_greeks.py --codes 3CN5 3IN5

# Resume from existing cache (skip already processed files)
python scripts/preprocess_greeks.py --codes 3CN5 --resume
```

### Performance Benefits

- **Faster Loading**: Avoids recalculating Black-Scholes formulas during training
- **Reliable Cache Control**: Default force reprocess ensures data consistency, optional --resume for speed
- **100% Processing Success Rate**: Fixed timestamp conversion and time matching issues
- **Cache Reuse**: Same option Greeks can be reused across different portfolios
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

## Dense Data Architecture (2025-08-21 Update)

The system now supports **dual data processing modes** to address training data sparsity issues:

### Data Modes Comparison

| Mode | Data Points | Training Sequences | Data Source | Use Case |
|------|-------------|-------------------|-------------|----------|
| **Sparse** | 46 | 1 | Option Greeks only | Legacy compatibility |
| **Dense** | 20,335+ | 19,836+ | Underlying + IV interpolation | Recommended for training |

### Dense Mode Technical Implementation

**Core Innovation**: Uses underlying asset data density (442x more points) with interpolated implied volatility to generate realistic option deltas at each underlying timestamp.

**Algorithm Flow**:
```
1. Load underlying data (USU5.npz): 20,335 price points
2. Load option IV data (sparse): 46 IV points  
3. Interpolate IV to underlying timestamps (linear/spline)
4. Calculate Black-Scholes delta at each underlying point
5. Generate sliding window training sequences (19,836 sequences)
```

**Performance Benefits**:
- **Training Stability**: Eliminates extreme loss swings (2.79 → 0.004)
- **Data Utilization**: 442x density increase vs sparse mode
- **Sequence Generation**: 19,836x more training sequences
- **Convergence**: Enables proper time-series learning patterns

### Technical Components

**New Modules**:
- `tools/vectorized_bs.py`: 19M+ calculations/second Black-Scholes
- Enhanced `data/precomputed_data_loader.py`: Dual-mode support
- `tests/test_underlying_dense_loader.py`: Comprehensive validation

**Integration Points**:
- `--underlying-dense-mode`: Command-line activation
- `--auto-dense-mode`: Automatic sparse→dense fallback (default: enabled)
- `dense_mode_params`: Configuration template support

### Usage Recommendations

**For Training**:
- Use dense mode for production training: `--underlying-dense-mode`
- Increase sequence length: `--sequence-length 1000` (vs 100 for sparse)
- Adjust batch size if needed: `--batch-size 16-32`

**Backward Compatibility**:
- All existing sparse mode workflows unchanged
- Auto-dense mode provides seamless upgrade path
- Original API preserved for legacy integrations

## Development Standards and Guidelines

### Code and Output Standards

**Emoji Usage Policy - STRICTLY PROHIBITED**:
- NEVER use emoji symbols in any program output, logs, or test results
- Emoji symbols cause conda environment execution failures
- Use text indicators instead: [PASS], [FAIL], [INFO], [WARN], [ERROR]
- This applies to all Python scripts, test files, and documentation generation

**Test File Management Standards**:
- ALL test scripts MUST be placed in the `tests/` directory
- Use descriptive names: `test_[component]_[functionality].py`
- After test completion, archive test files if they are temporary/experimental
- Keep only active, reusable test files in the tests directory
- Temporary test files should be moved to `tests/archive/` after completion

**File Organization Rules**:
- No temporary files in project root directory
- All experimental scripts go to appropriate subdirectories
- Clean up after task completion

### Example Correct Output Format:
```
[PASS] Test completed successfully
[FAIL] Error occurred: details here
[INFO] Processing data...
[WARN] Potential issue detected
[ERROR] Critical failure
```

### Test File Lifecycle:
1. Create test in `tests/test_[name].py`
2. Run and validate
3. If temporary/experimental: move to `tests/archive/`
4. If reusable: keep in `tests/` with proper documentation

## Task Management and Documentation Standards

### Mandatory Task Documentation Process

**For ALL tasks:**

1. **Task Initiation** - REQUIRED:
   - Create task document using template: `docs/templates/task_template.md`
   - Location: `docs/active/TASK_YYYY-MM-DD_brief_description.md`
   - Include all required sections: objectives, breakdown, risks, success criteria
   - Update immediately when task scope or timeline changes

2. **Progress Tracking** - REQUIRED:
   - Update task document Progress Log section after each significant step
   - Record work completed, issues encountered, next steps
   - Update task status and phase completion status in real-time
   - Minimum update frequency: after each major milestone

3. **Task Completion** - REQUIRED:
   - Complete final Progress Log entry with completion summary
   - Fill in "Lessons Learned" section
   - Update "Review and Approval" checklist
   - Archive task document to `docs/archive/completed_tasks/`
   - Update MASTER_DOCUMENTATION.md within 24 hours

### Task Documentation Trigger Points

Create task documentation for:
- Multi-step implementation tasks (>3 steps)
- System integration work
- New feature development
- Refactoring that affects multiple files
- Performance optimization projects
- Research and analysis tasks
- Bug fixing that requires investigation

### CLAUDE.local.md Integration

The local settings define:
- Task execution environment (conda environment 'learn')
- Testing procedures and commands
- Task recovery and resumption capabilities
- Progress persistence across interruptions

### Documentation Quality Standards

**Required Task Document Elements:**
- Clear objectives and success criteria
- Detailed phase breakdown with step-by-step progression
- Risk assessment and mitigation strategies
- Dependencies and technical specifications
- Real-time progress logging
- Testing plan and validation criteria

**Progress Update Standards:**
- Document after each significant milestone
- Include concrete work completed
- Record decision rationale
- Note issues and their resolutions
- Specify next actionable steps

### Compliance Requirements

**Mandatory Practices:**
- All complex tasks MUST have active task documents
- Progress updates MUST be real-time, not batched
- Task completion MUST include archival and master doc updates
- Documentation MUST follow established templates and standards

**Review Checkpoints:**
- Task initiation: Validate task document completeness
- Mid-task: Verify progress tracking is current
- Task completion: Confirm documentation and archival standards met

This ensures complete traceability, enables task resumption, and maintains project knowledge continuity.