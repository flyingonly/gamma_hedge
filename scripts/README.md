# Scripts Directory

This directory contains the main executable scripts for the Gamma Hedge project.

## Active Scripts

### Production Scripts

#### `run_production_training.py`
**Purpose**: Main production training interface for gamma hedging strategies
**Usage**:
```bash
# List available portfolios
python scripts/run_production_training.py --list-portfolios

# Get portfolio information
python scripts/run_production_training.py --portfolio-info single_atm_call

# Run training
python scripts/run_production_training.py --portfolio single_atm_call --epochs 50

# Custom training parameters
python scripts/run_production_training.py --portfolio single_atm_call \
    --epochs 100 --batch-size 64 --learning-rate 0.001 --disable-realtime-plots
```

**Features**:
- Option portfolio selection and management
- Real-time training monitoring and alerting
- Comprehensive visualization generation
- Professional-grade error handling and logging
- Results export and archival

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
**Purpose**: Batch preprocessing of options Greeks for performance optimization
**Usage**:
```bash
# Preprocess all available options
python scripts/preprocess_greeks.py

# Preprocess specific weekly codes
python scripts/preprocess_greeks.py --codes 3CN5 3IN5
```

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
- **Moved to `scripts/archive/`**: Deprecated testing and validation scripts
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

## Archived Scripts

Located in `scripts/archive/` - these are deprecated but preserved for reference:

- `run_single_option_hedge.py` - Superseded by production training pipeline
- `run_validation_fixed.py` - Temporary validation script
- `run_simple_validation.py` - Simplified validation script  
- `run_end_to_end_validation.py` - Comprehensive validation script

These archived scripts served their purpose during development but are no longer needed in the production workflow.

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