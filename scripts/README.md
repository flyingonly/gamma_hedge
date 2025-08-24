# Scripts Directory

This directory contains the main executable scripts for the Gamma Hedge project.

## Active Scripts

### Production Scripts

#### `run_production_training.py`
**Purpose**: Main production training interface for gamma hedging strategies using precomputed Greeks

**Core Architecture**:
- **Configuration Priority**: CLI Arguments > Portfolio Templates > TrainingConfig Defaults
- **Preprocessing Integration**: Works with preprocessed Greeks data from three modes
- **Portfolio Management**: Automatic option selection based on portfolio configurations
- **Time Series Support**: Daily-aligned training with variable-length sequences

**Basic Usage**:
```bash
# List available portfolio configurations
python scripts/run_production_training.py --list-portfolios

# Get detailed portfolio information  
python scripts/run_production_training.py --portfolio-info single_atm_call

# Basic training with default settings
python scripts/run_production_training.py --portfolio single_atm_call --epochs 50
```

**Advanced Training Workflows**:
```bash
# Time-series aligned training (recommended for production)
python scripts/run_production_training.py --portfolio single_atm_call \
    --epochs 50 --align-to-daily --split-ratios 0.6 0.3 0.1

# High-density preprocessing with custom parameters
python scripts/run_production_training.py --portfolio single_atm_call \
    --preprocessing-mode dense_interpolated --epochs 100 --batch-size 64

# Custom training configuration
python scripts/run_production_training.py --portfolio best_liquidity_single \
    --learning-rate 0.002 --min-daily-sequences 100 --disable-realtime-plots

# Development/testing with reduced visualization
python scripts/run_production_training.py --portfolio single_atm_call \
    --epochs 10 --batch-size 16 --disable-visualization
```

python scripts/run_production_training.py --portfolio single_atm_call --preprocessing-mode dense_daily_recalc

**Key Parameter Categories**:

**Portfolio Selection**:
- `--portfolio`: Portfolio configuration name (required)
- `--list-portfolios`: List all available portfolio configurations
- `--portfolio-info`: Show detailed portfolio information
- `--portfolio-config`: Path to custom portfolio configuration file

**Training Parameters** (Override portfolio defaults):
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Optimizer learning rate (default: 0.001)
- `--checkpoint-interval`: Save checkpoint every N epochs (default: 10)

**Time Series Configuration**:
- `--align-to-daily`: Enable daily-aligned sequences with variable lengths
- `--split-ratios`: Train/val/test split ratios (default: 0.7 0.2 0.1)
- `--min-daily-sequences`: Minimum data points per trading day (default: 50)
- `--sequence-length`: Fixed sequence length (default: 100, ignored if --align-to-daily)

**Data Processing**:
- `--preprocessing-mode`: Greeks preprocessing mode (sparse/dense_interpolated/dense_daily_recalc)

**System Configuration**:
- `--device`: Training device (cpu/cuda/auto, default: auto)
- `--seed`: Random seed for reproducibility (default: 42)
- `--disable-realtime-plots`: Disable real-time visualization plots
- `--log-level`: Logging verbosity (DEBUG/INFO/WARNING/ERROR)

**Configuration Priority System**:
1. **CLI Arguments** (highest priority) - Override any portfolio or default setting
2. **Portfolio Templates** - Define portfolio-specific defaults in `configs/portfolio_templates/`  
3. **TrainingConfig Defaults** - System defaults in `training/config.py`

**Output and Results**:
- Checkpoints saved to `checkpoints/` directory
- Training logs and metrics in `outputs/logs/`
- Visualization exports in `outputs/visualizations_*/`
- JSON results summary in `outputs/`

**Troubleshooting**:

*Issue*: `ValueError: num_samples should be a positive integer value, but got num_samples=0`  
*Solution*: This occurs with daily alignment on small datasets. Either disable `--align-to-daily` or use larger datasets with more trading days.

*Issue*: CUDA out of memory errors  
*Solution*: Reduce `--batch-size` (try 16 or 8) or use `--device cpu` for development.

*Issue*: No matching options found for portfolio  
*Solution*: Run preprocessing first: `python scripts/preprocess_greeks.py --mode dense_interpolated`

*Issue*: Slow training performance  
*Solution*: Use `--preprocessing-mode dense_interpolated` for better data density and training stability.

#### `run_tests.py`
**Purpose**: Unified test runner for the entire project
**Usage**:
```bash
# Run all tests
python scripts/run_tests.py --type all

# Run specific test categories
python scripts/run_tests.py --type unit
python scripts/run_tests.py --type integration
python scripts/run_tests.py --type functional

# Verbose output
python scripts/run_tests.py --type all --verbose
```

### Utility Scripts

#### `preprocess_greeks.py`
**Purpose**: Batch preprocessing of options Greeks with multiple calculation modes
**Usage**:
```bash
# Preprocess all available options (default: sparse mode)
python scripts/preprocess_greeks.py

# Preprocess with dense interpolated mode
python scripts/preprocess_greeks.py --mode dense_interpolated

# Preprocess with daily recalculation mode
python scripts/preprocess_greeks.py --mode dense_daily_recalc

# Preprocess specific weekly codes
python scripts/preprocess_greeks.py --codes 3CN5 3IN5 --mode dense_interpolated

# Resume from existing cache (skip already processed files)
python scripts/preprocess_greeks.py --codes 3CN5 --resume
```

**Preprocessing Modes (2025-08-22)**:
- `sparse`: Traditional option Greeks only
- `dense_interpolated`: High-density underlying data with IV interpolation
- `dense_daily_recalc`: Daily recalculation with gamma-based intraday updates

### Cleanup Scripts

#### `clean_outputs.bat` (Windows)
**Purpose**: Interactive cleanup of training outputs and model checkpoints
**Usage**:
```cmd
# From project root directory
scripts\clean_outputs.bat
```
**Features**:
- Interactive confirmation before deletion
- Detailed progress reporting
- Cleans outputs/, checkpoints/, and debug files
- Safe error handling

#### `clean_outputs.ps1` (PowerShell)
**Purpose**: PowerShell version of the cleanup script with enhanced UI
**Usage**:
```powershell
# From project root directory
.\scripts\clean_outputs.ps1
```
**Features**:
- Colored output for better readability
- File counting before cleanup
- Better error handling and warnings

#### `quick_clean.bat` (Windows)
**Purpose**: Silent, fast cleanup without confirmation prompts
**Usage**:
```cmd
# From project root directory
scripts\quick_clean.bat
```
**Features**:
- No user interaction required
- Silent operation (perfect for automation)
- Quick execution for development workflows

## Script Migration

All scripts have been reorganized from the project root to maintain a clean project structure:

- **Moved to `scripts/`**: Active, user-facing scripts
- **Removed**: Deprecated testing and validation scripts (were in `scripts/archive/`)
- **Deleted**: Debug and temporary testing scripts

### Path Updates

Scripts in the `scripts/` directory have been updated to correctly locate the project root:
```python
# Old (when in project root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# New (when in scripts/ subdirectory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
```

## Script Cleanup (2025-08-23)

The following deprecated scripts were removed during legacy code cleanup:

- `run_single_option_hedge.py` - Superseded by production training pipeline
- `run_validation_fixed.py` - Temporary validation script  
- `run_simple_validation.py` - Simplified validation script
- `run_end_to_end_validation.py` - Comprehensive validation script

These scripts contained outdated dependencies and were no longer compatible with the current architecture.

## Best Practices

1. **Always run scripts from the project root**:
   ```bash
   cd /path/to/gamma_hedge
   python scripts/run_production_training.py --help
   ```

2. **Use conda environment when available**:
   ```bash
   conda activate learn
   python scripts/run_production_training.py --portfolio single_atm_call
   ```

3. **Check script help for latest options**:
   ```bash
   python scripts/run_production_training.py --help
   python scripts/run_tests.py --help
   ```

## Integration with Project Workflow

- **Development**: Use `run_tests.py` for continuous testing
- **Production Training**: Use `run_production_training.py` for actual strategy development
- **Data Processing**: Use `preprocess_greeks.py` for data optimization
- **CI/CD**: Scripts are designed to work in automated environments

## Future Scripts

When adding new scripts:
1. Place in `scripts/` directory
2. Update path resolution for project root access
3. Follow naming convention: `verb_noun.py` (e.g., `run_training.py`, `process_data.py`)
4. Include comprehensive help text and argument parsing
5. Update this README with usage information