#!/usr/bin/env python3
"""
Initialize Project - Universal ML training project initializer
Auto-detect main script, analyze parameters using LLM, generate smart wrapper
"""

import os
import sys
import json
import glob
import re
import shutil
from datetime import datetime
from pathlib import Path

# Import LLM analyzer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_analyzer import smart_analyze_script


# =============================================================================
# Main Script Detection
# =============================================================================

def find_candidate_scripts(project_path: str) -> list:
    """Scan for candidate training scripts with priority ordering"""
    candidates = []
    priority = ['train.py', 'main.py', 'app.py', 'training.py', 'run.py', 
                'finetune.py', 'fine-tune.py', 'fit.py', 'experiment.py']
    
    # First check priority names
    for name in priority:
        path = os.path.join(project_path, name)
        if os.path.exists(path):
            candidates.append(path)
    
    # Then find other Python files that might be training scripts
    all_py = glob.glob(os.path.join(project_path, '*.py'))
    for path in all_py:
        if path not in candidates:
            # Check if it contains training-related keywords
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if any(kw in content.lower() for kw in ['train', 'epoch', 'fit', 'model']):
                    candidates.append(path)
            except:
                pass
    
    return candidates


def select_main_script(candidates: list) -> str:
    """Select the main training script from candidates"""
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]
    
    # Multiple candidates - let user choose
    print("\nMultiple candidate scripts found:")
    for i, path in enumerate(candidates, 1):
        print(f"  {i}. {os.path.basename(path)}")
    
    while True:
        try:
            choice = input("\nSelect main script (number, or 0 to specify path): ").strip()
            idx = int(choice)
            if idx == 0:
                custom = input("Enter script path: ").strip()
                return custom if os.path.exists(custom) else None
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        except ValueError:
            pass
        print("Invalid choice, please try again.")


# =============================================================================
# Smart Wrapper Generation
# =============================================================================

def generate_smart_wrapper(project_path: str, main_script: str, config: dict) -> str:
    """
    Generate intelligent wrapper that handles ALL parameter types
    
    Supports:
    - argparse: Adds --param arguments to command line
    - env_var: Sets environment variables before execution
    - hardcoded: Creates temporary modified script with injected values
    - config_file: Modifies config file before execution
    """
    params = config.get('parameters', {})
    
    # Categorize parameters
    argparse_params = []
    env_params = []
    hardcoded_params = []
    config_params = []
    
    for param_name, param_info in params.items():
        source = param_info.get('source', 'hardcoded')
        if source == 'argparse':
            argparse_params.append((param_name, param_info))
        elif source == 'env_var':
            env_params.append((param_name, param_info))
        elif source == 'hardcoded':
            hardcoded_params.append((param_name, param_info))
        elif source == 'config_file':
            config_params.append((param_name, param_info))
    
    # Prepare hardcoded params dict for JSON serialization
    
    # Generate environment variable setup
    env_setup_code = _generate_env_setup(env_params)
    
    # Generate argparse construction
    argparse_code = _generate_argparse_construction(argparse_params)
    
    # Generate hardcoded params dict as Python literal (not JSON) for safe insertion
    hardcoded_dict_lines = ["{"]
    for param_name, param_info in hardcoded_params:
        ptype = repr(param_info.get('type', 'str'))
        default = repr(param_info.get('default'))
        var_name = repr(param_info.get('var_name', param_name.upper()))
        source = repr(param_info.get('source', 'hardcoded'))
        modifiable = str(param_info.get('modifiable', True))
        hardcoded_dict_lines.append(f'        {repr(param_name)}: {{')
        hardcoded_dict_lines.append(f'            "type": {ptype},')
        hardcoded_dict_lines.append(f'            "default": {default},')
        hardcoded_dict_lines.append(f'            "var_name": {var_name},')
        hardcoded_dict_lines.append(f'            "source": {source},')
        hardcoded_dict_lines.append(f'            "modifiable": {modifiable}')
        hardcoded_dict_lines.append('        },')
    hardcoded_dict_lines.append("    }")
    hardcoded_dict_str = "\n".join(hardcoded_dict_lines)
    
    # Generate wrapper content
    wrapper = f'''#!/usr/bin/env python3
"""
Train Wrapper - Auto-generated by remote-training skill (Universal Mode)
Original script: {main_script}
Generated: {datetime.now().isoformat()}

Handles: argparse, env_var, hardcoded parameters
Captures: Loss, LR, Time statistics for enhanced reporting
"""

import os
import sys
import json
import subprocess
import re
import tempfile
import time
from pathlib import Path
from datetime import datetime

# Configuration paths
PROJECT_PATH = r"{project_path}"
MAIN_SCRIPT = r"{main_script}"
SCRIPT_PATH = os.path.join(PROJECT_PATH, MAIN_SCRIPT)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
RUN_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    """Load current configuration"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def setup_environment(config, current_values):
    """Setup environment variables for env_var parameters"""
{env_setup_code}

def build_argparse_args(config, current_values):
    """Build command line arguments for argparse parameters"""
    args = []
{argparse_code}
    return args

def create_modified_script(config, current_values):
    """
    Create temporary modified script with injected hardcoded parameters.
    Returns path to modified script or None if no hardcoded params.
    """
    hardcoded_params = {hardcoded_dict_str}
    
    if not hardcoded_params:
        return None
    
    print(f"[WRAPPER] Preparing to inject {{len(hardcoded_params)}} hardcoded parameter(s)", flush=True)
    
    # Read original script
    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        original_code = f.read()
    
    modified_code = original_code
    injected = []
    
    for param_name, param_info in hardcoded_params.items():
        new_value = current_values.get(param_name)
        if new_value is None:
            continue
        
        # Skip complex expressions that can't be safely injected
        original_default = param_info.get("default", "")
        if isinstance(original_default, str):
            # Check if original value is a complex Python expression (contains function calls, etc.)
            if any(c in original_default for c in ['(', ')', 'if', 'else', 'lambda', 'import']):
                print(f"[WRAPPER] Skipping {param_name}: complex expression (not safe to inject)", flush=True)
                continue
        
        var_name = param_info.get("var_name", param_name.upper())
        
        # Try to find and replace the variable assignment
        # Pattern 1: VAR_NAME = value (with optional type hint)
        patterns = [
            (rf"^(\\s*){{re.escape(var_name)}}(\\s*:\\s*\\w+)?\\s*=\\s*([^#\\n]+?)(\\s*(?:#|$))", f"\\1{{var_name}}\\2 = {{repr(new_value)}}\\3  # Injected by wrapper"),
            (rf"^(\\s*){{re.escape(var_name.lower())}}(\\s*:\\s*\\w+)?\\s*=\\s*([^#\\n]+?)(\\s*(?:#|$))", f"\\1{{var_name.lower()}}\\2 = {{repr(new_value)}}\\3  # Injected by wrapper"),
        ]
        
        for pattern, replacement_template in patterns:
            def replacer(match):
                return replacement_template.replace("\\1", match.group(1)).replace("\\2", match.group(2) or "").replace("\\3", match.group(4) or "")
            
            new_code, count = re.subn(pattern, replacer, modified_code, flags=re.MULTILINE | re.IGNORECASE)
            if count > 0:
                modified_code = new_code
                injected.append(f"{{var_name}} = {{new_value}}")
                break
    
    if not injected:
        print("[WRAPPER] No hardcoded parameters needed injection", flush=True)
        return None
    
    # Write modified script to temp location within project
    temp_dir = os.path.join(os.path.dirname(__file__), ".temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_script = os.path.join(temp_dir, MAIN_SCRIPT)
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(modified_code)
    
    print(f"[WRAPPER] Injected: {{', '.join(injected)}}", flush=True)
    return temp_script

def main():
    """Main wrapper execution"""
    # Force unbuffered output
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    print(f"[WRAPPER] Starting training wrapper", flush=True)
    print(f"[WRAPPER] Project: {{PROJECT_PATH}}", flush=True)
    print(f"[WRAPPER] Script: {{MAIN_SCRIPT}}", flush=True)
    print(f"[WRAPPER] Python: {{sys.executable}}", flush=True)
    print(f"[WRAPPER] Working dir: {{os.getcwd()}}", flush=True)
    print("[WRAPPER] Loading configuration...", flush=True)
    
    # Load configuration
    config = load_config()
    current_values = config.get("current_values", {{}})
    parameters = config.get("parameters", {{}})
    
    # Setup environment variables
    setup_environment(config, current_values)
    
    # Create modified script for hardcoded parameters
    modified_script = create_modified_script(config, current_values)
    
    # Change to project directory (CRITICAL for relative paths)
    original_cwd = os.getcwd()
    os.chdir(PROJECT_PATH)
    
    # Build command
    env = os.environ.copy()
    env["REMOTE_TRAINING"] = "1"
    env["CONFIG_PATH"] = CONFIG_PATH
    
    cmd = [sys.executable]
    
    if modified_script:
        # Use modified script - set PYTHONPATH to ensure imports work
        env["PYTHONPATH"] = PROJECT_PATH + os.pathsep + env.get("PYTHONPATH", "")
        cmd.append(modified_script)
    else:
        # Use original script
        cmd.append(SCRIPT_PATH)
    
    # Add argparse arguments
    cmd.extend(build_argparse_args(config, current_values))
    
    print(f"[WRAPPER] Command: {{' '.join(cmd)}}", flush=True)
    print(f"[WRAPPER] Working directory: {{os.getcwd()}}", flush=True)
    if modified_script:
        print(f"[WRAPPER] Using modified script with injected parameters", flush=True)
    print("[WRAPPER] Starting training process...", flush=True)
    print("=" * 50, flush=True)
    
    # Initialize statistics tracking
    stats = {{
        'start_time': datetime.now().isoformat(),
        'epochs': {{}},
        'total_epochs': 0,
        'final_lr': None
    }}
    
    # Run training with output capture for statistics
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Patterns for extracting metrics from output
        metric_patterns = {{
            'epoch': re.compile(r'(?:Epoch|epoch)\s*[\[\s:]?(\d+)', re.IGNORECASE),
            'loss': re.compile(r'(?:loss|train_loss|training loss)\s*[:=\s]+(\d+\.?\d*(?:e[+-]?\d+)?)', re.IGNORECASE),
            'val_loss': re.compile(r'(?:val_loss|validation_loss|test_loss|val loss)\s*[:=\s]+(\d+\.?\d*(?:e[+-]?\d+)?)', re.IGNORECASE),
            'lr': re.compile(r'(?:lr|learning_rate|learning rate)\s*[:=\s]+(\d+\.?\d*(?:e[+-]?\d+)?)', re.IGNORECASE),
            'total_epochs': re.compile(r'(?:total.?epoch|epoch.?count|num.?epoch)\s*[:=\s]*(\d+)', re.IGNORECASE),
        }}
        
        current_epoch = None
        epoch_start_time = time.time()
        
        # Read and process output line by line
        for line in process.stdout:
            # Print to stdout (pass through)
            print(line, end='', flush=True)
            
            line_lower = line.lower()
            
            # Detect epoch start
            epoch_match = metric_patterns['epoch'].search(line)
            if epoch_match:
                new_epoch = int(epoch_match.group(1))
                
                # Save previous epoch timing
                if current_epoch is not None and current_epoch in stats['epochs']:
                    elapsed = time.time() - epoch_start_time
                    stats['epochs'][current_epoch]['time'] = round(elapsed, 2)
                
                current_epoch = new_epoch
                epoch_start_time = time.time()
                if current_epoch not in stats['epochs']:
                    stats['epochs'][current_epoch] = {{}}
                stats['total_epochs'] = max(stats['total_epochs'], current_epoch)
            
            # Extract loss
            loss_match = metric_patterns['loss'].search(line)
            if loss_match and current_epoch is not None:
                try:
                    loss_val = float(loss_match.group(1))
                    stats['epochs'][current_epoch]['loss'] = loss_val
                except:
                    pass
            
            # Extract validation loss
            val_loss_match = metric_patterns['val_loss'].search(line)
            if val_loss_match and current_epoch is not None:
                try:
                    val_loss_val = float(val_loss_match.group(1))
                    stats['epochs'][current_epoch]['val_loss'] = val_loss_val
                except:
                    pass
            
            # Extract learning rate
            lr_match = metric_patterns['lr'].search(line)
            if lr_match:
                try:
                    lr_val = float(lr_match.group(1))
                    stats['final_lr'] = lr_val
                    if current_epoch is not None:
                        stats['epochs'][current_epoch]['lr'] = lr_val
                except:
                    pass
            
            # Extract total epochs from config print
            total_match = metric_patterns['total_epochs'].search(line)
            if total_match:
                try:
                    stats['total_epochs'] = int(total_match.group(1))
                except:
                    pass
        
        # Save final epoch timing
        if current_epoch is not None and current_epoch in stats['epochs']:
            elapsed = time.time() - epoch_start_time
            stats['epochs'][current_epoch]['time'] = round(elapsed, 2)
        
        process.wait()
        return_code = process.returncode
        
        # Save statistics to file
        stats['end_time'] = datetime.now().isoformat()
        stats['return_code'] = return_code
        
        # Find run directory - look for most recent run directory
        runs_dir = os.path.join(RUN_DIR, 'runs')
        stats_saved = False
        
        if os.path.exists(runs_dir):
            # Get all run directories sorted by name (timestamp)
            run_dirs = [d for d in os.listdir(runs_dir) 
                       if os.path.isdir(os.path.join(runs_dir, d))]
            if run_dirs:
                latest_run = sorted(run_dirs)[-1]
                stats_file = os.path.join(runs_dir, latest_run, 'wrapper_stats.json')
                try:
                    with open(stats_file, 'w', encoding='utf-8') as f:
                        json.dump(stats, f, indent=2, ensure_ascii=False)
                    print(f"[WRAPPER] Statistics saved to: {{stats_file}}", flush=True)
                    stats_saved = True
                except Exception as e:
                    print(f"[WRAPPER] Warning: Could not save stats: {{e}}", flush=True)
        
        if not stats_saved:
            # No fallback - runs/ directory must exist
            print(f"[WRAPPER] ERROR: Could not save stats to runs/ directory", flush=True)
            print(f"[WRAPPER] Make sure training is started via 'controller.py start'", flush=True)
        
    except Exception as e:
        print(f"[WRAPPER ERROR] {{e}}", flush=True)
        import traceback
        traceback.print_exc()
        return_code = 1
    finally:
        # Restore original directory
        os.chdir(original_cwd)
        
        # Cleanup temp file
        if modified_script and os.path.exists(modified_script):
            try:
                os.remove(modified_script)
            except:
                pass
    
    sys.exit(return_code)

if __name__ == "__main__":
    main()
'''
    
    return wrapper


def _generate_env_setup(env_params: list) -> str:
    """Generate environment variable setup code"""
    if not env_params:
        return "    pass  # No env_var parameters"
    
    lines = []
    for param_name, param_info in env_params:
        env_var = param_info.get('env_var', param_name.upper())
        lines.append(f'    os.environ["{env_var}"] = str(current_values.get("{param_name}", config["parameters"]["{param_name}"]["default"]))')
    
    return "\n".join(lines)


def _generate_argparse_construction(argparse_params: list) -> str:
    """Generate argparse argument construction code"""
    if not argparse_params:
        return "    pass  # No argparse parameters"
    
    lines = []
    for param_name, param_info in argparse_params:
        arg_name = param_info.get('arg_name', f"--{param_name.replace('_', '-')}")
        lines.append(f'    args.extend(["{arg_name}", str(current_values.get("{param_name}", config["parameters"]["{param_name}"]["default"]))])')
    
    return "\n".join(lines)


# =============================================================================
# Configuration File Generation
# =============================================================================

def generate_config(project_path: str, main_script: str, analysis_result: dict, 
                   conda_env: str, conda_recommendation: dict = None) -> dict:
    """Generate configuration from analysis result"""
    remote_dir = os.path.join(project_path, 'remote_training')
    
    config = {
        'project_path': project_path,
        'main_script': os.path.basename(main_script),
        'remote_dir': remote_dir,
        'wrapper_type': 'smart',
        'conda_env': conda_env,
        'conda_recommendation': conda_recommendation or {},
        'parameters': analysis_result.get('parameters', {}),
        'current_values': {},
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    # Extract current values from parameters
    for param_name, param_info in config['parameters'].items():
        config['current_values'][param_name] = param_info.get('default')
    
    return config


def generate_parameters_md(config: dict) -> str:
    """Generate human-readable PARAMETERS.md"""
    params = config.get('parameters', {})
    conda_env = config.get('conda_env', 'base')
    conda_rec = config.get('conda_recommendation', {})
    
    lines = [
        "# Training Configuration Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Project: {config.get('project_path', 'N/A')}",
        f"Main Script: {config.get('main_script', 'N/A')}",
        f"Conda Environment: `{conda_env}`",
        "",
    ]
    
    # Add conda recommendation info if available
    if conda_rec:
        rec_env = conda_rec.get('recommended', 'N/A')
        confidence = conda_rec.get('confidence', 'N/A')
        reasoning = conda_rec.get('reasoning', '')
        frameworks = conda_rec.get('detected_frameworks', [])
        imports = conda_rec.get('detected_imports', [])
        
        lines.extend([
            "## Conda Environment Analysis",
            "",
            f"**Selected Environment:** `{conda_env}`",
            "",
            f"**LLM Recommendation:** `{rec_env}` (confidence: {confidence})",
            "",
            f"**Detected Frameworks:** {', '.join(frameworks) if frameworks else 'None'}",
            "",
            f"**Key Imports:** {', '.join(imports[:10]) if imports else 'None'}",
            "",
            f"**Reasoning:** {reasoning}",
            "",
        ])
    
    lines.extend([
        f"## Detected Parameters ({len(params)} total)",
        "",
        "| Parameter | Type | Default | Source | Modifiable |",
        "|-----------|------|---------|--------|------------|",
    ])
    
    for param_name, param_info in sorted(params.items()):
        ptype = param_info.get('type', 'str')
        default = param_info.get('default', 'N/A')
        source = param_info.get('source', 'unknown')
        modifiable = 'YES' if param_info.get('modifiable', True) else 'NO'
        
        # Format default for display
        if isinstance(default, str) and len(str(default)) > 20:
            default = str(default)[:17] + '...'
        
        lines.append(f"| {param_name} | {ptype} | {default} | {source} | {modifiable} |")
    
    lines.extend([
        "",
        "## Usage",
        "",
        "Modify parameters in `config.json` or use:",
        "```bash",
        "python controller.py update --param <name> --value <value>",
        "```",
        "",
        "Then start training:",
        "```bash",
        "python controller.py start",
        "```",
    ])
    
    return "\n".join(lines)


# =============================================================================
# Main Initialization
# =============================================================================

def find_available_conda_envs():
    """Find all available conda environments on the system"""
    import subprocess
    envs = []
    
    # Try to get conda env list
    try:
        result = subprocess.run(
            ['conda', 'env', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if parts:
                    env_name = parts[0]
                    envs.append(env_name)
    except Exception:
        pass
    
    return envs


def select_conda_environment(recommendation: dict, available_envs: list, user_specified: str = None) -> str:
    """
    Select the conda environment to use.
    Priority: user_specified > LLM_recommendation > default
    """
    print("\n" + "="*60)
    print("CONDA ENVIRONMENT SELECTION")
    print("="*60)
    
    # If user specified, validate and use it
    if user_specified:
        if user_specified in available_envs:
            print(f"[USER] Using specified environment: {user_specified}")
            return user_specified
        else:
            print(f"[WARN] Specified environment '{user_specified}' not found in available envs")
            print(f"       Available: {', '.join(available_envs) if available_envs else 'None detected'}")
    
    # Display LLM recommendation
    if recommendation:
        rec_env = recommendation.get('recommended', 'base')
        confidence = recommendation.get('confidence', 'low')
        reasoning = recommendation.get('reasoning', 'No reasoning provided')
        alternatives = recommendation.get('alternatives', [])
        frameworks = recommendation.get('detected_frameworks', [])
        imports = recommendation.get('detected_imports', [])
        
        print(f"\n[LLM ANALYSIS] Recommended Environment: {rec_env}")
        print(f"               Confidence: {confidence.upper()}")
        print(f"\n[DETECTED FRAMEWORKS] {', '.join(frameworks) if frameworks else 'None'}")
        print(f"[KEY IMPORTS] {', '.join(imports[:10]) if imports else 'None'}...")
        print(f"\n[REASONING] {reasoning}")
        
        if alternatives:
            print(f"\n[ALTERNATIVES] {', '.join(alternatives)}")
        
        # Check if recommended exists
        if rec_env in available_envs:
            print(f"\n✓ Recommended environment '{rec_env}' is AVAILABLE")
            use_rec = input(f"\nUse recommended environment '{rec_env}'? [Y/n]: ").strip().lower()
            if use_rec in ('', 'y', 'yes'):
                return rec_env
        else:
            print(f"\n✗ Recommended environment '{rec_env}' is NOT available")
    
    # Let user choose from available
    if available_envs:
        print(f"\nAvailable conda environments:")
        print(f"  0. base (default)")
        for i, env in enumerate(available_envs, 1):
            marker = " ← RECOMMENDED" if recommendation and env == recommendation.get('recommended') else ""
            print(f"  {i}. {env}{marker}")
        
        while True:
            choice = input("\nSelect environment (number or name): ").strip()
            
            # Numeric choice
            if choice.isdigit():
                idx = int(choice)
                if idx == 0:
                    return "base"
                if 1 <= idx <= len(available_envs):
                    return available_envs[idx - 1]
            
            # Name choice
            if choice in available_envs:
                return choice
            
            # Allow custom name even if not in list
            if choice:
                confirm = input(f"Environment '{choice}' not found. Use it anyway? [y/N]: ").strip().lower()
                if confirm in ('y', 'yes'):
                    return choice
            
            print("Invalid choice, please try again.")
    else:
        print("\n[WARN] Could not detect conda environments")
        env = input("Enter conda environment name (or 'base'): ").strip()
        return env if env else "base"


def init_project(project_path: str, force: bool = False, analyze_only: bool = False, 
                 use_llm: bool = True, conda_env: str = None):
    """
    Initialize a project for remote training
    
    Args:
        project_path: Path to the training project
        force: Force re-initialization if already initialized
        analyze_only: Only analyze and print results, don't create files
        use_llm: Try to use LLM for analysis
        conda_env: User-specified conda environment (optional)
    """
    project_path = os.path.abspath(project_path)
    
    print(f"[SCAN] Project: {project_path}")
    
    # Check if already initialized
    remote_dir = os.path.join(project_path, 'remote_training')
    if os.path.exists(remote_dir) and not force and not analyze_only:
        print(f"[WARN] Project already initialized at: {remote_dir}")
        print("       Use --force to re-initialize")
        return False
    
    # Step 1: Find candidate scripts
    candidates = find_candidate_scripts(project_path)
    print(f"[FILE] Found {len(candidates)} candidate script(s): {[os.path.basename(c) for c in candidates]}")
    
    if not candidates:
        print("[ERR] No training scripts found")
        print("       Looking for: train.py, main.py, or any .py with training keywords")
        return False
    
    # Step 2: Select main script
    main_script_path = select_main_script(candidates)
    if not main_script_path:
        print("[ERR] No main script selected")
        return False
    
    main_script = os.path.basename(main_script_path)
    print(f"[MAIN] Selected: {main_script}")
    
    # Step 3: Analyze script with LLM
    print(f"[ANLZ] Analyzing {main_script}...")
    try:
        analysis = smart_analyze_script(main_script_path, use_llm=use_llm)
    except Exception as e:
        print(f"[ERR] Analysis failed: {e}")
        return False
    
    # Step 4: Display results
    params = analysis.get('parameters', {})
    conda_recommendation = analysis.get('conda_environment', {})
    
    print(f"\n[OK] Analysis complete!")
    print(f"     Total parameters: {len(params)}")
    
    # Group by source
    by_source = {}
    for name, info in params.items():
        source = info.get('source', 'unknown')
        by_source.setdefault(source, []).append(name)
    
    for source, names in by_source.items():
        print(f"     [{source}]: {len(names)} params")
    
    # Step 4.5: Find and select conda environment
    available_envs = find_available_conda_envs()
    selected_env = select_conda_environment(conda_recommendation, available_envs, conda_env)
    
    if analyze_only:
        print("\n[ANALYZE-ONLY] Analysis complete. No files created.")
        print("               Run without --analyze-only to initialize project.")
        return True
    
    # Step 5: Create directory structure
    print(f"\n[INIT] Creating remote_training directory...")
    os.makedirs(remote_dir, exist_ok=True)
    os.makedirs(os.path.join(remote_dir, 'runs'), exist_ok=True)
    
    # Step 6: Generate configuration
    print("[CONF] Generating config.json...")
    config = generate_config(project_path, main_script_path, analysis, selected_env, conda_recommendation)
    
    config_path = os.path.join(remote_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Step 7: Generate PARAMETERS.md
    print("[DOC] Generating PARAMETERS.md...")
    param_md = generate_parameters_md(config)
    param_md_path = os.path.join(remote_dir, 'PARAMETERS.md')
    with open(param_md_path, 'w', encoding='utf-8') as f:
        f.write(param_md)
    
    # Step 8: Generate smart wrapper script
    print("[WRAP] Generating train_wrapper.py...")
    wrapper = generate_smart_wrapper(project_path, main_script, config)
    wrapper_path = os.path.join(remote_dir, 'train_wrapper.py')
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper)
    
    # Step 9: Initialize parameter history
    print("[HIST] Initializing parameter history...")
    history = {
        'entries': [{
            'timestamp': datetime.now().isoformat(),
            'action': 'init',
            'parameters': config['current_values'].copy()
        }]
    }
    history_path = os.path.join(remote_dir, 'params_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    # Step 10: Completion
    print(f"\n{'='*60}")
    print("INITIALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nRemote directory: {remote_dir}")
    print(f"\nFiles created:")
    print(f"  - {config_path}")
    print(f"  - {param_md_path}")
    print(f"  - {wrapper_path}")
    print(f"  - {history_path}")
    print(f"\nNext steps:")
    print(f"  1. Review parameters:  python controller.py show")
    print(f"  2. Update parameters:  python controller.py update --param <name> --value <val>")
    print(f"  3. Start training:     python controller.py start")
    print(f"{'='*60}")
    
    return True


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize ML training project for remote control')
    parser.add_argument('project_path', help='Path to training project')
    parser.add_argument('--force', action='store_true', help='Force re-initialization')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not create files')
    parser.add_argument('--no-llm', action='store_true', help='Use regex analysis only (no LLM)')
    parser.add_argument('--conda-env', help='Specify conda environment (bypasses auto-detection)')
    
    args = parser.parse_args()
    
    success = init_project(
        args.project_path,
        force=args.force,
        analyze_only=args.analyze_only,
        use_llm=not args.no_llm,
        conda_env=args.conda_env
    )
    
    sys.exit(0 if success else 1)
