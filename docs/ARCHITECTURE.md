# Gamma Hedge System Architecture

**Version**: 1.0  
**Last Updated**: 2025-08-25  
**Status**: Post-Modular Refactor  

## Overview

The Gamma Hedge System is a machine learning-based optimal execution platform that combines reinforcement learning with options Greeks calculation for sophisticated trading strategy optimization. The system implements policy gradient methods to learn optimal trade execution timing while minimizing market impact costs.

## Architecture Principles

### Core Design Philosophy

1. **KISS (Keep It Simple, Stupid)**: Simple, understandable solutions over complex ones
2. **YAGNI (You Aren't Gonna Need It)**: Implement only currently required features  
3. **SOLID Principles**: Maintainable, extensible, and testable code
4. **Unified Configuration**: Single source of truth for all system parameters
5. **Backward Compatibility**: Smooth migration during architecture evolution

### Modular Design Goals

- **Separation of Concerns**: Clear boundaries between components
- **Interface Standardization**: Consistent APIs across modules
- **Configuration Unification**: Centralized parameter management
- **Dependency Minimization**: Reduced coupling between components

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gamma Hedge System                      │
├─────────────────────────────────────────────────────────────┤
│  Entry Points                                               │
│  ├── scripts/run_production_training.py (Main CLI)          │
│  ├── main.py (Legacy Interface)                            │
│  └── scripts/preprocess_greeks.py (Data Preprocessing)      │
├─────────────────────────────────────────────────────────────┤
│  Core Layer (core/)                                         │
│  ├── config.py          │ Unified Configuration System     │
│  ├── interfaces.py      │ Standard Data Interfaces        │
│  └── orchestrator.py    │ Workflow Coordination           │
├─────────────────────────────────────────────────────────────┤
│  Training Layer (training/)                                 │
│  ├── production_trainer.py │ Production Training Workflow  │
│  ├── trainer.py           │ Core Training Engine           │
│  ├── training_monitor.py  │ Metrics & Monitoring           │
│  └── engine.py           │ Unified Training Engine         │
├─────────────────────────────────────────────────────────────┤
│  Data Layer (data/)                                         │
│  ├── Portfolio Management                                   │
│  │   ├── option_portfolio_manager.py │ Portfolio Selection │
│  │   └── portfolio.py               │ Portfolio Interface  │
│  ├── Data Loading                                          │
│  │   ├── data_loader.py             │ Unified Data Loading │
│  │   ├── precomputed_data_loader.py │ Greeks Data Loader   │
│  │   └── market_simulator.py        │ Market Simulation    │
│  ├── Data Processing                                        │
│  │   ├── greeks_preprocessor.py     │ Options Greeks Calc  │
│  │   ├── processors.py              │ Data Processors      │
│  │   └── loaders.py                 │ Data Loaders         │
│  └── Data Utilities                                         │
│      └── data_types.py              │ Data Type Definitions │
├─────────────────────────────────────────────────────────────┤
│  Model Layer (models/)                                      │
│  ├── policy_network.py              │ Neural Network Model │
│  └── value_function_network.py      │ Value Function Model │
├─────────────────────────────────────────────────────────────┤
│  Common Utilities (common/)                                 │
│  ├── collate.py         │ Data Batching Functions          │
│  └── exceptions.py      │ Custom Exception Classes         │
├─────────────────────────────────────────────────────────────┤
│  Tools & Utilities (tools/, utils/)                         │
│  ├── black_scholes.py   │ Options Pricing & Greeks         │
│  ├── config.py          │ Legacy Configuration Utils       │
│  └── policy_tracker.py  │ Policy Decision Tracking         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Core Layer (`core/`)

The foundation layer providing unified interfaces and configuration management.

#### Configuration System (`core/config.py`)
- **Unified Configuration**: Single source of truth for all parameters
- **Priority Hierarchy**: CLI Arguments > Environment Variables > Config Files > Defaults
- **Type Safety**: Dataclass-based configuration with validation
- **Backward Compatibility**: Legacy configuration support during migration

```python
@dataclass
class Config:
    market: MarketConfig
    training: TrainingConfig  
    model: ModelConfig
    delta_hedge: DeltaHedgeConfig
    greeks: GreeksConfig
```

**Configuration Classes**:
- `MarketConfig`: Market simulation parameters
- `TrainingConfig`: Learning rates, epochs, batch sizes, regularization
- `ModelConfig`: Neural network architecture
- `DeltaHedgeConfig`: Options-specific parameters  
- `GreeksConfig`: Greeks preprocessing configuration

#### Interfaces (`core/interfaces.py`)
- **TrainingData**: Standard training data format
- **DataResult**: Backward compatibility data format  
- **EvaluationResult**: Standardized evaluation results
- **PortfolioPosition**: Portfolio position representation

#### Orchestrator (`core/orchestrator.py`)
- **Workflow Coordination**: Manages training, evaluation, preprocessing workflows
- **Resource Management**: Device setup, reproducibility, cleanup
- **Error Handling**: Comprehensive error handling and recovery
- **Progress Tracking**: Workflow status and history management

### 2. Training Layer (`training/`)

Handles all aspects of model training and evaluation.

#### Production Trainer (`training/production_trainer.py`)
- **Complete Training Workflow**: Portfolio selection → Data preparation → Training → Results
- **Configuration Integration**: Unified configuration with priority handling
- **Monitoring Integration**: Real-time training metrics and visualization
- **Professional Output**: Structured results and comprehensive logging

#### Core Trainer (`training/trainer.py`)
- **Policy Gradient Training**: REINFORCE with baseline and advantage functions
- **Value Function Integration**: Dual network training (policy + value)
- **Regularization**: Entropy regularization and execution cost penalties
- **Flexible Architecture**: Supports various tensor shapes and data formats

#### Training Monitor (`training/training_monitor.py`)
- **Real-time Metrics**: Loss, hedge ratios, execution costs
- **Visualization**: Training progress plots and decision tracking
- **Session Management**: Training session logging and persistence
- **Performance Analytics**: Training efficiency and convergence analysis

### 3. Data Layer (`data/`)

Comprehensive data management and processing pipeline.

#### Data Loading Architecture

**Unified Data Loading** (`data/data_loader.py`):
- **Multi-source Support**: Market simulation, precomputed Greeks, real data
- **Variable-length Sequences**: Daily-aligned sequences with attention masks
- **Collate Functions**: Sophisticated batching for different sequence lengths
- **Configuration Integration**: Seamless integration with unified configuration

**Precomputed Greeks** (`data/precomputed_data_loader.py`):
- **High-performance Loading**: Optimized NPZ file loading
- **Multi-portfolio Support**: Simultaneous loading of multiple option positions
- **Time-series Alignment**: Proper timestamp handling and sequence creation
- **Memory Efficiency**: Only loads required date ranges and options

#### Portfolio Management

**Option Portfolio Manager** (`data/option_portfolio_manager.py`):
- **Template-based Selection**: Pre-configured portfolio templates
- **Dynamic Discovery**: Automatic option scanning and validation
- **Selection Algorithms**: Best liquidity, ATM selection, spread strategies
- **Reporting**: Comprehensive portfolio analysis and statistics

#### Data Processing Pipeline

```
Raw Data Sources
    │
    ├── Options Data (Excel/CSV)
    ├── Underlying Data (Market prices)
    └── Greeks Calculations (Black-Scholes)
    │
    ▼
Greeks Preprocessor
    ├── sparse: Basic Greeks calculation
    ├── dense_interpolated: High-density with interpolation  
    └── dense_daily_recalc: Daily base + intraday updates
    │
    ▼
NPZ Storage (Cached)
    │
    ▼
Precomputed Data Loader
    ├── Portfolio position weighting
    ├── Time-series sequence creation
    └── Daily alignment with variable lengths
    │
    ▼
Training Data Loader
    ├── Batch creation with collate functions
    ├── Device placement (CPU/GPU)
    └── Attention mask generation
```

### 4. Model Layer (`models/`)

Neural network implementations for policy and value functions.

#### Policy Network (`models/policy_network.py`)
- **Architecture**: Configurable feedforward network
- **Input**: State representation (prices, positions, time features)
- **Output**: Execution probability (0-1) via sigmoid activation
- **Regularization**: Dropout, batch normalization support

#### Value Function Network (`models/value_function_network.py`)
- **Purpose**: Baseline estimation for variance reduction
- **Architecture**: Shared input processing with policy network
- **Output**: State value estimation
- **Integration**: Seamless integration with policy gradient training

### 5. Preprocessing System

#### Greeks Calculation Pipeline

**Three Preprocessing Modes**:

1. **Sparse Mode** (`preprocessing_mode: sparse`)
   - Basic Greeks calculation from option data
   - Lightweight processing
   - Suitable for prototyping and testing

2. **Dense Interpolated Mode** (`preprocessing_mode: dense_interpolated`)
   - High-density underlying data integration
   - Implied volatility interpolation  
   - Recommended for training

3. **Dense Daily Recalc Mode** (`preprocessing_mode: dense_daily_recalc`)
   - Daily base calculation with intraday gamma updates
   - Most realistic trading scenario simulation
   - Variable-length sequences (54-276 timesteps)

**Preprocessing Architecture**:
```python
class GreeksPreprocessor:
    def preprocess_batch(self, weekly_codes, mode):
        for code in weekly_codes:
            if mode == "sparse":
                return self._process_sparse(code)
            elif mode == "dense_interpolated": 
                return self._process_dense_interpolated(code)
            elif mode == "dense_daily_recalc":
                return self._process_dense_daily_recalc(code)
```

## Data Flow Architecture

### Training Data Flow

```
Portfolio Selection
    │ (OptionPortfolioManager)
    ▼
Option Discovery & Validation
    │ (Available options scanning)
    ▼
Greeks Data Loading  
    │ (PrecomputedGreeksDataset)
    ▼
Sequence Generation
    │ (Daily-aligned, variable-length)
    ▼
Batch Creation
    │ (Variable-length collate function)
    ▼
Tensor Processing
    │ (Shape normalization, attention masks)
    ▼
Model Training
    │ (Policy gradient with value baseline)
    ▼
Results & Visualization
```

### Configuration Flow

```
CLI Arguments
    │ (Highest priority)
    ▼
Environment Variables
    │ (GAMMA_HEDGE_*)
    ▼
Configuration Files
    │ (JSON format)
    ▼
Portfolio Templates
    │ (Template-specific overrides)
    ▼
Default Values
    │ (Dataclass defaults)
    ▼
Unified Configuration Object
    │ (Type-safe, validated)
    ▼
Component Initialization
```

## Key Technical Features

### 1. Variable-Length Sequence Handling

**Challenge**: Different trading days have different amounts of data
**Solution**: Advanced collate function with attention masks

```python
def variable_length_collate_fn(batch):
    # Normalize tensor shapes: [1, seq_len, features] -> [seq_len, features]
    processed_tensors = []
    for tensor in batch:
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        processed_tensors.append(tensor)
    
    # Pad to same length with attention masks
    padded_tensors = pad_sequence(processed_tensors, batch_first=True)
    attention_mask = create_attention_mask(padded_tensors, original_lengths)
    
    return DataResult(padded_tensors, metadata={'attention_mask': attention_mask})
```

### 2. Unified Configuration System

**Benefits**:
- Single source of truth for all parameters
- Clear priority hierarchy
- Type safety and validation
- Easy parameter inheritance and overrides

**Usage Example**:
```python
# CLI overrides portfolio overrides defaults
config = create_config(cli_args={'epochs': 100, 'batch_size': 64})

# Access configuration
model = PolicyNetwork(
    input_dim=config.model.input_dim,
    hidden_dims=config.model.hidden_dims,
    dropout_rate=config.model.dropout_rate
)
```

### 3. Portfolio-Based Training

**Templates** (`configs/portfolio_templates/`):
- `single_atm_call.json`: Basic single option training
- `single_atm_put.json`: Put option variant
- `best_liquidity_single.json`: Highest quality data selection
- `call_spread.json`: Multi-option strategies (future)

**Selection Algorithm**:
1. Load portfolio template configuration
2. Discover available options matching criteria
3. Apply selection algorithm (best liquidity, ATM, etc.)
4. Validate data quality and completeness
5. Generate position weights and training sequences

### 4. Advanced Training Features

**Policy Gradient Enhancements**:
- **Returns-to-go**: Better reward attribution
- **Value Function Baseline**: Variance reduction
- **Advantage Function**: A(s,a) = Q(s,a) - V(s)
- **Entropy Regularization**: Exploration encouragement

**Execution Cost Modeling**:
```python
execution_cost = base_cost + variable_cost * |trade_size|
total_cost = market_impact + execution_cost + opportunity_cost
reward = -total_cost  # Minimize total trading cost
```

## Performance Characteristics

### Preprocessing Performance

| Mode | Data Points | Processing Time | Memory Usage | Disk Space |
|------|-------------|----------------|--------------|------------|
| Sparse | 117 | ~1 second | Low | 280KB |
| Dense Interpolated | ~5,000 | ~10 seconds | Medium | ~1MB |
| Dense Daily Recalc | 20,335 | ~30 seconds | High | ~2MB |

### Training Performance

| Configuration | Sequence Length | Batch Size | Training Speed | Memory |
|--------------|----------------|------------|----------------|---------|
| Sparse | 52 | 16 | 1.6s/epoch | 1GB |
| Dense Daily Recalc | 54-276 | 16 | 8.0s/epoch | 2GB |
| Dense Interpolated | 100-200 | 32 | ~15s/epoch | 3GB |

### Model Architecture

**Default Policy Network**:
- Input dimension: 4 (prev_holding, current_holding, price, time)
- Hidden layers: [128, 64, 32]
- Parameters: 11,009 total
- Output: Single execution probability

## Error Handling and Recovery

### Comprehensive Error Management

1. **Configuration Validation**: Parameter consistency and constraint checking
2. **Data Quality Checks**: Missing data, corrupted files, format validation  
3. **Model Compatibility**: Input dimension matching, device compatibility
4. **Training Stability**: NaN detection, gradient clipping, checkpoint recovery
5. **Resource Management**: Memory monitoring, GPU availability, disk space

### Recovery Mechanisms

- **Checkpoint Resume**: Automatic training resumption from last successful checkpoint
- **Graceful Degradation**: Fallback to CPU if GPU unavailable
- **Data Validation**: Skip corrupted data files with logging
- **Parameter Fallback**: Default values for missing configuration parameters

## Future Architecture Considerations

### Planned Enhancements

1. **Multi-Asset Support**: Expand beyond single option to portfolio of assets
2. **Real-time Data Integration**: Live market data feeds
3. **Distributed Training**: Multi-GPU and multi-node training support
4. **Advanced Strategies**: Spread trading, volatility surface modeling
5. **Risk Management**: VaR calculation, position sizing, drawdown limits

### Scalability Roadmap

- **Data Pipeline**: Streaming data processing for real-time training
- **Model Serving**: Production deployment with low-latency inference
- **Cloud Integration**: AWS/GCP deployment with auto-scaling
- **Monitoring Dashboard**: Real-time system health and performance metrics

---

**Document Version**: 1.0  
**Architecture Status**: ✅ Stable (Post-Refactor)  
**Next Review**: 2025-09-15  
**Maintainer**: Development Team