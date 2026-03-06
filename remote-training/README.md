# Remote Training Skill

**Universal ML Training Controller** - Zero code changes required. Works with any Python training script.

## 🚀 Quick Start (3 Steps)

```bash
# 1. Initialize your project
python scripts/controller.py init /path/to/your/project

# 2. Start training (opens window for real-time monitoring)
python scripts/controller.py start

# 3. Check results
python scripts/controller.py status
python scripts/controller.py report
```

## ✨ Key Features

- **🔮 LLM-Powered Analysis** - Automatically detects ALL training parameters
- **🎯 Universal Support** - Works with argparse, env_var, and **hardcoded** parameters
- **💉 Auto-Injection** - Modifies hardcoded values on-the-fly without changing your code
- **📊 Experiment Tracking** - Each run isolated with logs, metrics, and reports
- **🔄 Hot Parameter Updates** - Change configs without restarting the skill

## 📖 Example Usage

### Modify Any Parameter

```bash
# Works with argparse parameters
python controller.py update epochs 20

# Works with environment variables  
python controller.py update learning_rate 0.0005

# Works with HARDCODED constants too!
python controller.py update hidden_dim 512
python controller.py update sequence_length 1024
```

### Compare Experiments

```bash
# View history
python controller.py history

# Compare two runs
python controller.py compare 0 5

# Rollback if needed
python controller.py rollback initial
```

## 🔧 How Hardcoded Injection Works

Your original code stays 100% unchanged:

```python
# Your train.py (unchanged)
HIDDEN_DIM = 256
SEQUENCE_LENGTH = 512
```

The skill automatically:
1. Creates temporary modified copy
2. Injects new values
3. Runs training
4. Cleans up

```python
# Temporary injected version
HIDDEN_DIM = 512  # Injected by wrapper
SEQUENCE_LENGTH = 1024  # Injected by wrapper
```

## 📁 Project Structure After Init

```
your_project/
├── train.py                    # Your code (unchanged!)
└── remote_training/            # All skill files here
    ├── config.json            # Parameter configuration
    ├── PARAMETERS.md          # Human-readable docs
    ├── train_wrapper.py       # Auto-generated wrapper
    └── runs/                  # Experiment records
        └── 20260303_123456_run1/
            ├── train.log      # Training output
            └── summary.md     # Auto-generated report
```

## 🎓 Complete Command Reference

### Initialization
```bash
python controller.py init /path/to/project
python controller.py init /path/to/project --analyze-only  # Preview only
python controller.py init /path/to/project --no-llm        # Regex only
```

### Training
```bash
python controller.py start                    # New window (recommended)
python controller.py start --no-window        # Current terminal
python controller.py start --override epochs=20 lr=0.001
python controller.py stop
python controller.py status
```

### Configuration
```bash
python controller.py show                     # View current config
python controller.py update <param> <value>   # Single parameter
python controller.py update-batch lr=0.001 batch_size=64
python controller.py history                  # View change history
python controller.py rollback                 # Rollback to initial
```

### Logs & Reports
```bash
python controller.py logs                     # View latest logs
python controller.py logs -n 100              # Last 100 lines
python controller.py logs --run <run_id>      # Specific run
python controller.py report                   # Generate report
```

## 🤖 LLM Support

The skill automatically uses (in priority order):
1. **OpenAI GPT-4** - Set `OPENAI_API_KEY` env var
2. **Local Ollama** - Runs on localhost:11434
3. **Regex Fallback** - Works without any API

## 🛠️ Requirements

- Python 3.7+
- No external dependencies for basic usage
- Optional: `openai` package for GPT-4 analysis
- Optional: Ollama for local LLM analysis

## 🐛 Troubleshooting

**"No training scripts found"**
→ Name your script `train.py`, `main.py`, or any `.py` with training code

**"Parameter not found"**
→ Run `python controller.py show` to see detected parameters

**LLM analysis fails**
→ Use `--no-llm` flag for regex-only analysis

## 📄 License

MIT License - Free to use and modify!

## 💡 Pro Tips

1. **Always use `--window` mode** on Windows to see real-time progress
2. **Check `PARAMETERS.md`** after init to see what was detected
3. **Use `update-batch`** for multiple changes at once
4. **Reports auto-generate** when training completes

---

**Made for ML engineers who want to iterate fast without code pollution! 🚀**
