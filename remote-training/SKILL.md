# Remote Training Skill

**Universal ML training controller** - Works with any Python training script (PyTorch, TensorFlow, JAX, etc.)

## Overview

This skill provides **zero-code-change** remote control for ML training:
- **LLM-powered analysis** automatically detects all training parameters
- **Universal wrapper generation** handles argparse, env_var, and hardcoded parameters
- **Bypass mode** keeps your original project code 100% unchanged
- **Auto-injection** modifies hardcoded parameters on-the-fly

## Quick Start

```bash
# 1. Initialize project (LLM analyzes code, detects parameters AND recommends Conda environment)
python scripts/controller.py init /path/to/project

# 2. Review detected parameters and Conda environment
cat remote_training/PARAMETERS.md

# 3. Start training (automatically uses selected Conda environment)
python scripts/controller.py start

# 4. Check status
python scripts/controller.py status
```

## ⚠️ CRITICAL: Correct Training Workflow

**ALWAYS use `controller.py start` to launch training.**

### ❌ WRONG - Never do this:
```bash
# NEVER run wrapper directly - this bypasses critical setup!
python remote_training/train_wrapper.py
```

**What goes wrong:**
- `runs/` directory is **NOT created**
- No run ID is generated
- Training stats save to `wrapper_stats_fallback.json` instead of proper run directory
- No `summary.md` report is generated
- Process tracking is broken

### ✅ CORRECT - Always do this:
```bash
# Use controller to start training - handles all setup
python scripts/controller.py start
```

**What happens correctly:**
- Creates `runs/<timestamp>_runN/` directory
- Generates unique run ID
- Copies wrapper to run directory
- Saves stats to proper location
- Generates `summary.md` report
- Full process tracking

### 🔁 Complete First-Time Workflow

When running on a **new machine** or **fresh project clone**:

```bash
# Step 1: Initialize (creates remote_training/ folder)
python scripts/init_project.py /path/to/project --conda-env pytorch

# Step 2: Verify configuration
python scripts/controller.py show

# Step 3: Update parameters if needed
python scripts/controller.py update epochs 12

# Step 4: Start training using controller (NEVER run wrapper directly!)
python scripts/controller.py start

# Step 5: Check progress
python scripts/controller.py status

# Step 6: View report after completion
python scripts/controller.py report
```

### ⚡ One-Liner for Repeated Runs

After initial setup, you can combine steps:
```bash
# Update and start in one go
python scripts/controller.py update epochs 20 && python scripts/controller.py start
```

### 📝 Important Notes

1. **First run must use `init_project.py`** - This creates the entire `remote_training/` infrastructure
2. **Never run `train_wrapper.py` directly** - Always use `controller.py start`
3. **Wrapper is auto-generated** - Don't modify it manually; re-run `init_project.py` if needed
4. **Each `start` creates a new run** - Previous runs are preserved in `runs/` directory
5. **Reports are auto-generated** - Only when using `controller.py start`

## Report Language

All generated reports (`summary.md`) are in **English** for consistency and compatibility.

## How It Works

### 🧠 Conda Environment Intelligence (NEW!)

The skill now includes **intelligent Conda environment detection and recommendation**:

**How it works:**
1. **Code Analysis**: LLM scans imports, framework usage, and library dependencies
2. **Framework Detection**: Identifies PyTorch, TensorFlow, JAX, Transformers, etc.
3. **Smart Recommendation**: Suggests the most appropriate Conda environment
4. **User Choice**: You can accept the recommendation or specify your own

**Decision Priority:**
1. **User-specified** (`--conda-env pytorch`) → Always use this
2. **LLM Recommendation** → If available and user accepts
3. **Interactive Selection** → Choose from detected environments
4. **Default fallback** → Use `base` environment

**Example Scenarios:**

| Project Type | Detected Imports | Recommended Env | Confidence |
|--------------|------------------|-----------------|------------|
| PyTorch + CUDA | `torch`, `torchvision`, `cuda` | `pytorch` | High |
| TensorFlow 2.x | `tensorflow`, `keras` | `tensorflow` | High |
| Transformers NLP | `transformers`, `datasets` | `transformers` | Medium |
| JAX/Flax | `jax`, `flax`, `optax` | `jax` | Medium |
| Sklearn Classic | `sklearn`, `numpy`, `pandas` | `base` or `ml` | Low |

### Universal Parameter Detection

The skill analyzes your training script and detects parameters from ANY source:

| Source | Detection Method | Modification Strategy |
|--------|-----------------|----------------------|
| `argparse` | `parser.add_argument()` | Adds `--param value` to command line |
| `env_var` | `os.environ.get()` | Sets environment variables |
| `hardcoded` | `VAR = value` | Creates temporary modified script |
| `config_file` | `yaml/json.load()` | Modifies config before execution |
| `dataclass` | `@dataclass` | Updates dataclass fields |

### Intelligent Wrapper Generation

After initialization, the skill generates `train_wrapper.py` that:

1. **Reads config.json** for current parameter values
2. **Categorizes parameters** by their source type
3. **Applies modifications** using the appropriate strategy:
   - argparse → Command line arguments
   - env_var → Environment variables  
   - hardcoded → Code injection (creates temp modified script)
4. **Executes training** with all modifications applied
5. **Cleans up** temporary files after execution

## Installation

```bash
# Copy skill to your nanobot workspace
cp -r remote-training ~/.nanobot/workspace/skills/

# Or use clawhub (if available)
clawhub install remote-training
```

## Commands

### Initialization

```bash
# Standard initialization with LLM analysis AND Conda environment detection
python scripts/controller.py init /path/to/project

# Specify Conda environment directly (bypasses auto-detection)
python scripts/controller.py init /path/to/project --conda-env pytorch

# Preview analysis without creating files
python scripts/controller.py init /path/to/project --analyze-only

# Force re-initialization
python scripts/controller.py init /path/to/project --force

# Use regex only (faster, no LLM API needed, skips Conda recommendations)
python scripts/controller.py init /path/to/project --no-llm
```

**Conda Environment Selection Flow:**

When you run `init`, the skill will:
1. Analyze your code to detect frameworks (PyTorch, TensorFlow, etc.)
2. Recommend the best Conda environment with detailed reasoning
3. Show available environments on your system
4. Let you accept the recommendation, choose another, or specify manually

Example interaction:
```
============================================================
CONDA ENVIRONMENT SELECTION
============================================================

[LLM ANALYSIS] Recommended Environment: pytorch
               Confidence: HIGH

[DETECTED FRAMEWORKS] pytorch, torchvision
[KEY IMPORTS] torch, torch.nn, torch.optim, torchvision...

[REASONING] The script imports torch and torch.nn extensively,
            uses CUDA device detection (torch.cuda.is_available()),
            and implements a custom Transformer architecture.
            The 'pytorch' environment is specifically designed
            for PyTorch-based deep learning projects.

[ALTERNATIVES] torch, deep-learning

✓ Recommended environment 'pytorch' is AVAILABLE

Use recommended environment 'pytorch'? [Y/n]: 
```

### Metric Detection During Initialization

During initialization, the skill analyzes your training script to understand what metrics it outputs. This helps ensure the report captures all available data:

**Detected Output Patterns:**
- Print statements with accuracy metrics
- Loss logging statements
- Learning rate scheduling outputs
- Timing/duration prints

**Example:**
```python
# In your train.py - these will be detected
print(f"Epoch:{epoch} Accuracy on test:{test_acc:.2f}")
print(f"Loss: {loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
```

The LLM analyzer identifies these patterns and documents them in `PARAMETERS.md` for reference.

### Training Control

**Window Modes:**

| Mode | Command | Behavior |
|------|---------|----------|
| **New Window** (default) | `start` | Opens new CMD window showing real-time training progress |
| **Current Terminal** | `start --no-window` | Runs in current terminal, blocks until complete |
| **Background Silent** | `start --silent` | Runs silently in background, no window |

```bash
# Start training (DEFAULT: opens new CMD window for visibility)
python scripts/controller.py start

# Start with parameter overrides
python scripts/controller.py start --override epochs=20 lr=0.001

# Run in current terminal (no new window)
python scripts/controller.py start --no-window

# Run silently in background (no window, no output) - only when explicitly requested
python scripts/controller.py start --silent

# Stop training
python scripts/controller.py stop

# Check status
python scripts/controller.py status

# View logs
python scripts/controller.py logs
python scripts/controller.py logs -n 100
python scripts/controller.py logs --run 20260303_123456_run1
```

### Parameter Management

```bash
# Show current configuration
python scripts/controller.py show

# Update single parameter
python scripts/controller.py update epochs 20
python scripts/controller.py update learning_rate 0.0005 --note "Reduced for fine-tuning"

# Update multiple parameters
python scripts/controller.py update-batch lr=0.001 batch_size=64 epochs=50

# View change history
python scripts/controller.py history
python scripts/controller.py history -n 20

# Compare history entries
python scripts/controller.py compare
python scripts/controller.py compare 0 5

# Rollback to initial configuration
python scripts/controller.py rollback

# Rollback to specific entry
python scripts/controller.py rollback 3
```

### Report Generation

```bash
# Generate report for latest run
python scripts/controller.py report

# Generate report for specific run
python scripts/controller.py report --run 20260303_123456_run1

# Force regenerate report (overwrite existing)
python scripts/controller.py report --force
```

**Report Contents (`summary.md`):**
- **Overview** - Run ID, timestamps, duration, status, total epochs
- **Configuration Parameters** - Values with source type annotations (CLI arg / env var / hardcoded)
- **Best Performance** - Highest Test/Cross/Train accuracy, Lowest Loss/Val Loss
- **Training Statistics** - Max/Min/Average/Final values, improvement range
- **Epoch Details** - Detailed metrics table for each epoch
- **Trend Visualization** - ASCII charts for accuracy and loss trends
- **Raw Training Log** - Collapsible full log content
- **Errors and Warnings** - All detected issues and warnings
- **Related Files** - Quick access to other files

Reports are automatically generated when training completes.

### Auto-Detected Metrics in Epoch Table

The `summary.md` report includes an **Epoch Details** table with the following columns. Metrics are automatically extracted from your training log using pattern matching - only metrics found in your log will be populated.

| Column | Description | Detection Patterns | Availability |
|--------|-------------|-------------------|--------------|
| `Epoch` | Training epoch number | `Epoch 1`, `Epoch [1/10]` | **Always** |
| `Test Acc` | Test/validation set accuracy | `test accuracy: 95.5`, `Accuracy on test: 95.5%` | Detected if printed |
| `Cross Acc` | Cross-validation accuracy | `cross accuracy`, `val accuracy`, `validation acc` | Detected if printed |
| `Train Acc` | Training set accuracy | `train accuracy`, `training acc` | Detected if printed |
| `Loss` | Training loss | `loss: 0.234`, `train_loss: 0.234`, `Loss: 0.5` | Detected if printed |
| `Val Loss` | Validation/test loss | `val_loss`, `validation_loss`, `test_loss` | Detected if printed |
| `LR` | Learning rate | `lr: 0.001`, `learning_rate: 1e-4` | Detected if printed |
| `Time` | Epoch duration | `time: 45.2s`, `duration: 10.5 sec` | Detected if printed |

**Legend:** `-` in the table means the metric was not detected in the training log.

### How to Ensure Metrics Are Captured

To have all metrics populated in the report, ensure your training script prints them in a recognizable format:

```python
# Recommended format (will be auto-detected)
print(f"Epoch:{epoch} Accuracy on test:{test_acc:.2f}, Accuracy on cross:{cross_acc:.2f}")
print(f"Train Acc: {train_acc:.2f}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
print(f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")

# Alternative formats also supported
print(f"Epoch [{epoch}/{total}], Test Acc: {test_acc}%, Val Acc: {val_acc}%")
print(f"loss={loss:.4f}, val_loss={val_loss:.4f}, lr={lr}")
```

**Note:** The parser uses regex pattern matching and is case-insensitive. Values can be percentages (95.5%) or decimals (0.955).

## Generated File Structure

```
project/
├── train.py                    # Original code (100% unchanged)
└── remote_training/            # Bypass folder (all skill files here)
    ├── config.json            # Current parameter configuration + Conda env
    ├── PARAMETERS.md          # Human-readable parameter documentation
    ├── params_history.json    # Parameter change history
    ├── train_wrapper.py       # LLM-generated universal wrapper
    └── runs/                  # Training run records
        └── 20260303_123456_run1/
            ├── config_snapshot.json   # Config at run start
            ├── train.log              # Training output log
            ├── training.pid           # Process ID
            ├── summary.md             # Auto-generated report
            ├── monitor.py             # Background monitor
            └── run_training.bat       # Windows launcher
```

### config.json Fields

| Field | Description |
|-------|-------------|
| `project_path` | Absolute path to project directory |
| `main_script` | Entry point Python file |
| `remote_dir` | Path to remote_training folder |
| `wrapper_type` | Wrapper generation mode (smart) |
| `conda_env` | **Selected Conda environment name** |
| `conda_recommendation` | **LLM analysis results** (recommended env, confidence, reasoning, detected frameworks) |
| `parameters` | Detected training parameters with metadata |
| `current_values` | Current parameter values for next run |

## Configuration File (config.json)

```json
{
  "project_path": "/path/to/project",
  "main_script": "train.py",
  "remote_dir": "/path/to/project/remote_training",
  "wrapper_type": "smart",
  "conda_env": "pytorch",
  "conda_recommendation": {
    "recommended": "pytorch",
    "confidence": "high",
    "reasoning": "Script uses torch.nn, torch.optim, and CUDA detection extensively",
    "alternatives": ["torch", "deep-learning"],
    "detected_frameworks": ["pytorch", "torchvision"],
    "detected_imports": ["torch", "torch.nn", "torch.optim", "torchvision"]
  },
  "parameters": {
    "epochs": {
      "type": "int",
      "default": 10,
      "description": "Number of training epochs",
      "source": "argparse",
      "arg_name": "--epochs",
      "modifiable": true
    },
    "learning_rate": {
      "type": "float",
      "default": 0.001,
      "description": "Learning rate for optimizer",
      "source": "env_var",
      "env_var": "LEARNING_RATE",
      "var_name": "LR",
      "modifiable": true
    },
    "hidden_dim": {
      "type": "int",
      "default": 256,
      "description": "Hidden dimension",
      "source": "hardcoded",
      "var_name": "HIDDEN_DIM",
      "modifiable": true
    }
  },
  "current_values": {
    "epochs": 10,
    "learning_rate": 0.001,
    "hidden_dim": 256
  }
}
```

## Supported Parameter Types

The skill automatically detects and handles:

### Training Hyperparameters
- `epochs`, `batch_size`, `learning_rate`, `weight_decay`
- `warmup_steps`, `gradient_accumulation_steps`, `max_steps`

### Model Architecture
- `hidden_dim`, `num_layers`, `num_heads`, `dropout`
- `d_model`, `vocab_size`, `num_classes`, `num_channels`
- `sequence_length`, `max_seq_len`, `context_window`

### Optimizer Settings
- `optimizer`, `scheduler`, `momentum`, `beta1`, `beta2`
- `lr_scheduler`, `gamma`, `step_size`, `patience`

### Data Processing
- `train_split`, `num_workers`, `pin_memory`
- `data_path`, `dataset_path`, `augmentation`

### Hardware/Runtime
- `device`, `num_gpus`, `fp16`, `bf16`, `seed`

### Paths
- `save_dir`, `log_dir`, `checkpoint_dir`, `output_dir`

## Parameter Modification Examples

### 1. Argparse Parameters

**Original code:**
```python
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
```

**Skill handles:**
```bash
python controller.py update epochs 20
# Wrapper generates: python train.py --epochs 20 --lr 0.001
```

### 2. Environment Variable Parameters

**Original code:**
```python
EPOCHS = int(os.environ.get('EPOCHS', 10))
LR = float(os.environ.get('LEARNING_RATE', 0.001))
```

**Skill handles:**
```bash
python controller.py update epochs 20
# Wrapper sets: os.environ['EPOCHS'] = '20'
```

### 3. Hardcoded Parameters (Auto-Injection!)

**Original code:**
```python
HIDDEN_DIM = 256
SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
```

**Skill handles:**
```bash
python controller.py update hidden_dim 512
python controller.py update sequence_length 1024
```

**What happens:**
1. Wrapper reads original `train.py`
2. Creates temporary copy with injected values:
   ```python
   HIDDEN_DIM = 512  # Injected by wrapper
   SEQUENCE_LENGTH = 1024  # Injected by wrapper
   BATCH_SIZE = 32
   ```
3. Executes modified script
4. Cleans up after training

## LLM Analysis

The skill uses LLM (GPT-4, local models, or regex fallback) to:

1. **Read entire script** - Not just pattern matching, but true code comprehension
2. **Identify entry points** - Finds `main()`, `train()`, or top-level code
3. **Extract all parameters** - Even nested in functions or classes
4. **Detect source types** - Determines if argparse/env_var/hardcoded
5. **🔥 Recommend Conda environment** - Analyzes imports and frameworks to suggest the best environment
6. **Generate recommendations** - Suggests improvements for parameter handling

### Conda Environment Detection

The LLM analyzes your code to recommend the most suitable Conda environment:

**Detection Criteria:**

| Signals | Example | Likely Environment |
|---------|---------|-------------------|
| PyTorch imports | `import torch`, `from torch import nn` | `pytorch`, `torch` |
| TensorFlow/Keras | `import tensorflow as tf` | `tensorflow`, `tf` |
| Transformers | `from transformers import *` | `transformers`, `huggingface` |
| JAX ecosystem | `import jax`, `import flax` | `jax`, `flax` |
| Scientific ML | `sklearn`, `numpy`, `scipy` | `base`, `ml`, `scipy` |
| Deep Learning | `pytorch` + `cuda` usage | `pytorch`, `cuda` |

**Confidence Levels:**
- **High** - Clear framework imports with specific version patterns
- **Medium** - Multiple indicators but some ambiguity
- **Low** - Generic imports or mixed frameworks

### LLM Providers

The skill automatically tries (in order):
1. **OpenAI** - If `OPENAI_API_KEY` is set
2. **Local LLM** - If Ollama is running on localhost:11434
3. **Regex Fallback** - Enhanced pattern matching

## Error Handling

The skill includes robust error handling:

- **Invalid parameter names** - Shows available parameters
- **Type mismatches** - Validates and converts values
- **Missing files** - Clear error messages with suggestions
- **Training failures** - Preserves logs for debugging
- **Permission errors** - Graceful fallbacks

## Troubleshooting

### "No training scripts found"
- Ensure your project has a `.py` file with training keywords
- Or specify the script manually during initialization

### "Parameter not found"
- Run `python controller.py show` to see available parameters
- Check that the parameter is defined in your training script

### "Wrapper injection failed"
- Hardcoded variable names must match exactly (case-sensitive)
- Check `PARAMETERS.md` for detected variable names

### LLM analysis fails
- Set `OPENAI_API_KEY` environment variable, or
- Install and run Ollama locally, or
- Use `--no-llm` flag for regex-only analysis

### Conda Environment Issues

**"Conda environment not found"**
```bash
# Check available environments
conda env list

# Create a new environment if needed
conda create -n pytorch python=3.11 pytorch torchvision -c pytorch

# Re-initialize with correct environment
python scripts/controller.py init . --force --conda-env pytorch
```

**"Wrong environment selected"**
```bash
# View current config
cat remote_training/config.json | grep conda_env

# Update conda environment
# Edit config.json and change "conda_env" field, or
# Re-initialize with --conda-env flag
python scripts/controller.py init . --force --conda-env correct_env
```

**"LLM recommended wrong environment"**
- Use `--conda-env` flag to override: `python scripts/controller.py init . --conda-env myenv`
- The recommendation is just a suggestion - you always have final control
- Consider improving your code's import patterns to help LLM detection

## Advanced Usage

### Custom Parameter Mapping

If the skill doesn't detect a parameter correctly, you can manually edit `config.json`:

```json
{
  "parameters": {
    "my_custom_param": {
      "type": "int",
      "default": 100,
      "description": "Custom parameter",
      "source": "hardcoded",
      "var_name": "MY_CUSTOM_VAR",
      "modifiable": true
    }
  }
}
```

### Multi-Project Setup

You can initialize multiple projects:

```bash
python scripts/controller.py init /path/to/project1
python scripts/controller.py init /path/to/project2
```

Each project gets its own `remote_training/` folder.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Commands                                               │
│  init → start → update → stop → status                      │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  controller.py (Unified Entry Point)                         │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
┌──────────────┐ ┌──────────┐ ┌─────────────┐
│ init_project │ │run_training│ │config_manager│
└──────┬───────┘ └─────┬────┘ └──────┬──────┘
       │               │             │
       ↓               ↓             ↓
┌──────────────┐ ┌──────────┐ ┌─────────────┐
│llm_analyzer  │ │auto_reporter│ │  History   │
└──────────────┘ └──────────┘ └─────────────┘
```

## File Reference

| File | Responsibility |
|------|---------------|
| `scripts/controller.py` | Unified CLI entry point |
| `scripts/init_project.py` | Project initialization, wrapper generation |
| `scripts/llm_analyzer.py` | LLM-based parameter extraction |
| `scripts/run_training.py` | Training execution management |
| `scripts/config_manager.py` | Parameter updates, history, rollback |
| `scripts/auto_reporter.py` | Report generation |
| `train_wrapper.py` | Auto-generated universal wrapper |

## License

MIT License - Feel free to use and modify.

## Contributing

Contributions welcome! Areas for improvement:
- Additional LLM providers (Anthropic, local models)
- More parameter patterns
- Better error recovery
- GUI interface
- Remote server support
