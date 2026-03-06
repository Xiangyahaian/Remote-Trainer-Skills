#!/usr/bin/env python3
"""
Run Training - Universal training execution manager
Features:
- Each run has its own isolated directory
- Auto-detects parameter types and applies them correctly
- Real-time log monitoring
- Automatic report generation on completion
"""

import os
import sys
import json
import subprocess
import signal
import time
import re
from datetime import datetime
from pathlib import Path


class TrainingRunner:
    """Universal training runner supporting all parameter types"""
    
    def __init__(self):
        self.remote_dir = self._find_remote_dir()
        if not self.remote_dir:
            print("[ERR] remote_training directory not found")
            print("       Please run: python controller.py init <project_path>")
            sys.exit(1)
        
        self.config_path = os.path.join(self.remote_dir, 'config.json')
        self.runs_dir = os.path.join(self.remote_dir, 'runs')
        
        # Ensure runs directory exists
        os.makedirs(self.runs_dir, exist_ok=True)
        
        # Load config
        self.config = self._load_config()
    
    def _find_remote_dir(self) -> str:
        """Find remote_training directory"""
        current = os.getcwd()
        for _ in range(5):
            remote = os.path.join(current, 'remote_training')
            if os.path.exists(remote):
                return remote
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return None
    
    def _find_conda_executable(self) -> str:
        """Find conda executable"""
        # Common conda locations on Windows
        possible_paths = [
            os.path.expanduser('~/anaconda3/Scripts/conda.exe'),
            os.path.expanduser('~/miniconda3/Scripts/conda.exe'),
            os.path.expanduser('~/anaconda3/condabin/conda.bat'),
            os.path.expanduser('~/miniconda3/condabin/conda.bat'),
            f'C:\\Users\\{os.environ.get("USERNAME", "")}\\anaconda3\\Scripts\\conda.exe',
            f'C:\\Users\\{os.environ.get("USERNAME", "")}\\miniconda3\\Scripts\\conda.exe',
            f'C:\\Users\\{os.environ.get("USERNAME", "")}\\anaconda3\\condabin\\conda.bat',
            f'C:\\Users\\{os.environ.get("USERNAME", "")}\\miniconda3\\condabin\\conda.bat',
            # Also try where command
        ]
        
        for path in possible_paths:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded
        
        # Try 'where conda'
        try:
            result = subprocess.run(['where', 'conda'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                conda_path = result.stdout.strip().split('\n')[0]
                if os.path.exists(conda_path):
                    return conda_path
        except:
            pass
        
        return None
    
    def _find_conda_python(self, env_name: str) -> str:
        """Find Python executable in conda environment"""
        # Common conda locations
        possible_paths = [
            # Windows
            os.path.expanduser(f'~/anaconda3/envs/{env_name}/python.exe'),
            os.path.expanduser(f'~/miniconda3/envs/{env_name}/python.exe'),
            f'C:/Users/{os.environ.get("USERNAME", "")}/anaconda3/envs/{env_name}/python.exe',
            f'C:/Users/{os.environ.get("USERNAME", "")}/miniconda3/envs/{env_name}/python.exe',
            # Linux/Mac
            os.path.expanduser(f'~/anaconda3/envs/{env_name}/bin/python'),
            os.path.expanduser(f'~/miniconda3/envs/{env_name}/bin/python'),
            f'/opt/conda/envs/{env_name}/bin/python',
        ]
        
        for path in possible_paths:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded
        
        # Try to find via conda command
        try:
            result = subprocess.run(
                ['conda', 'run', '-n', env_name, 'python', '-c', 'import sys; print(sys.executable)'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                python_path = result.stdout.strip()
                if os.path.exists(python_path):
                    return python_path
        except:
            pass
        
        return None
    
    def _load_config(self) -> dict:
        """Load configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _is_running(self) -> bool:
        """Check if training is running by checking latest run"""
        runs = sorted(os.listdir(self.runs_dir)) if os.path.exists(self.runs_dir) else []
        if not runs:
            return False
        
        latest_run = runs[-1]
        pid_file = os.path.join(self.runs_dir, latest_run, 'training.pid')
        
        if not os.path.exists(pid_file):
            return False
        
        try:
            with open(pid_file, 'r') as f:
                content = f.read().strip()
                # Skip window mode markers
                if content.startswith('window:'):
                    return False
                pid = int(content)
            
            if os.name == 'nt':
                result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                       capture_output=True, text=True)
                return str(pid) in result.stdout
            else:
                os.kill(pid, 0)
                return True
        except (ValueError, OSError, ProcessLookupError):
            return False
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        existing = [d for d in os.listdir(self.runs_dir) if d.startswith(timestamp)]
        run_num = len(existing) + 1
        return f"{timestamp}_run{run_num}"
    
    def _create_run_directory(self, run_id: str) -> str:
        """Create directory structure for a run"""
        run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def _save_run_snapshot(self, run_dir: str, run_id: str, log_file: str):
        """Save configuration snapshot for this run"""
        snapshot = {
            'run_id': run_id,
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'current_values': self.config.get('current_values', {}),
            'parameter_types': self.config.get('parameter_types', {}),
            'log_file': log_file
        }
        
        snapshot_path = os.path.join(run_dir, 'config_snapshot.json')
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
    
    def start(self, window_mode: bool = True, overrides: dict = None, silent_mode: bool = False):
        """Start training"""
        if self._is_running():
            print("[WARN] Training is already running")
            print("       Run 'python controller.py status' to check")
            return
        
        # Generate run ID and create directory
        run_id = self._generate_run_id()
        run_dir = self._create_run_directory(run_id)
        
        # Create log file
        log_path = os.path.join(run_dir, 'train.log')
        
        # Save configuration snapshot
        self._save_run_snapshot(run_dir, run_id, log_path)
        
        # Apply parameter overrides if any
        current_values = self.config.get('current_values', {}).copy()
        if overrides:
            current_values.update(overrides)
            # Save updated config
            self.config['current_values'] = current_values
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"[START] Starting training...")
        print(f"   Run ID: {run_id}")
        print(f"   Log: {log_path}")
        print(f"   Params: {current_values}")
        
        # Find wrapper script
        wrapper_path = os.path.join(self.remote_dir, 'train_wrapper.py')
        if not os.path.exists(wrapper_path):
            print(f"[ERR] Wrapper not found: {wrapper_path}")
            return
        
        if window_mode and os.name == 'nt':
            self._start_window_mode(run_dir, run_id, log_path, wrapper_path)
        else:
            self._start_direct_mode(run_dir, run_id, log_path, wrapper_path)
    
    def _start_window_mode(self, run_dir: str, run_id: str, log_path: str, wrapper_path: str):
        """Start training with a new CMD window for real-time monitoring (Windows)"""
        # Detect conda environment from config or use default
        conda_env = self.config.get('conda_env', 'pytorch')
        project_path = self.config.get('project_path', os.getcwd())
        
        # Get epochs for display
        epochs_display = self.config.get('current_values', {}).get('epochs', 'N/A')
        
        # Get skill directory for auto_reporter
        skill_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Find conda python - CRITICAL: Use the conda env specified in config
        conda_python = self._find_conda_python(conda_env)
        if not conda_python:
            print(f"[WARN] Could not find conda env '{conda_env}', trying fallback...")
            conda_python = sys.executable
        else:
            print(f"[INFO] Using conda env: {conda_env}")
            print(f"[INFO] Python: {conda_python}")
        
        # Use the conda python for execution
        python_exe = conda_python
        
        # Escape backslashes for Python string literals in the generated code
        log_file_escaped = log_path.replace('\\', '\\\\')
        wrapper_escaped = wrapper_path.replace('\\', '\\\\')
        remote_dir_escaped = self.remote_dir.replace('\\', '\\\\')
        run_dir_escaped = run_dir.replace('\\', '\\\\')
        skill_dir_escaped = skill_dir.replace('\\', '\\\\')
        project_path_escaped = project_path.replace('\\', '\\\\')
        conda_python_escaped = conda_python.replace('\\', '\\\\')
        python_exe_escaped = python_exe.replace('\\', '\\\\')
        pid_file_escaped = os.path.join(run_dir, 'training.pid').replace('\\', '\\\\')
        
        # Create monitor script that captures output and displays it in real-time
        monitor_code = f'''#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import re
import traceback

log_file = r"{log_file_escaped}"
wrapper = r"{wrapper_escaped}"
remote_dir = r"{remote_dir_escaped}"
run_id_display = "{run_id}"
run_dir = r"{run_dir_escaped}"
skill_dir = r"{skill_dir_escaped}"
project_path = r"{project_path_escaped}"
conda_python = r"{conda_python_escaped}"
python_exe = r"{python_exe_escaped}"

print("=" * 60)
print(f"Training Monitor - {{run_id_display}}")
print("=" * 60)
print(f"Run ID: {{run_id_display}}")
print(f"Epochs: {epochs_display}")
print(f"Python: {{python_exe}}")
print("=" * 60)
print()

# Verify wrapper exists
if not os.path.exists(wrapper):
    print(f"[ERR] Wrapper not found: {{wrapper}}")
    print("Press any key to exit...")
    input()
    sys.exit(1)

try:
    # Open log file for writing
    with open(log_file, 'w', encoding='utf-8') as log_f:
        # Start training process with unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        
        print(f"[INFO] Starting training process...")
        print(f"[INFO] Command: {{python_exe}} -u {{wrapper}}")
        print()
        
        process = subprocess.Popen(
            [python_exe, '-u', wrapper],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=remote_dir,
            env=env,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Save PID
        pid_file = r"{pid_file_escaped}"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"[INFO] Process started (PID: {{process.pid}})")
        print()
        
        # Read and display output in real-time
        completion_detected = False
        line_count = 0
        try:
            for line in process.stdout:
                line_count += 1
                # Write to log
                log_f.write(line)
                log_f.flush()
                
                # Print to console (handle encoding issues for Windows)
                try:
                    print(line, end='', flush=True)
                except UnicodeEncodeError:
                    print(line.encode('ascii', errors='ignore').decode('ascii'), end='', flush=True)
                
                # Check for training completion (all epochs done)
                if 'Epoch' in line and '/' in line:
                    match = re.search(r'Epoch\\s*\\d+.*?(\\d+)\\s*[/:]\\s*(\\d+)', line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        if current >= total and not completion_detected:
                            completion_detected = True
                            print("\\n[INFO] Training completion detected, generating summary...")
                            # Auto-generate summary
                            try:
                                sys.path.insert(0, skill_dir)
                                from auto_reporter import generate_summary
                                success, msg = generate_summary(run_dir)
                                print(f"[AUTO] {{msg}}")
                            except Exception as e:
                                print(f"[WARN] Auto-summary failed: {{e}}")
        
        except KeyboardInterrupt:
            print("\\n[WARN] Interrupted by user")
            process.terminate()
        
        process.wait()
        
        print(f"\\n[INFO] Total lines processed: {{line_count}}")
        
        # Final summary generation (if not already done)
        if not completion_detected:
            print("\\n[INFO] Generating final summary...")
            try:
                sys.path.insert(0, skill_dir)
                from auto_reporter import generate_summary
                success, msg = generate_summary(run_dir)
                print(f"[AUTO] {{msg}}")
            except Exception as e:
                print(f"[WARN] Auto-summary failed: {{e}}")
        
        print()
        print("=" * 60)
        if process.returncode == 0:
            print("[OK] Training completed successfully!")
        else:
            print(f"[ERR] Training failed (code: {{process.returncode}})")
        print("=" * 60)
        print(f"Log saved: {{log_file}}")
        print(f"Summary: {{run_dir}}\\summary.md")

except Exception as e:
    print()
    print("=" * 60)
    print("[ERR] Monitor encountered an error:")
    print(f"      {{e}}")
    print()
    traceback.print_exc()
    print("=" * 60)

finally:
    print()
    print("Press any key to exit...")
    try:
        input()
    except:
        pass
'''
        
        monitor_script = os.path.join(run_dir, 'monitor.py')
        with open(monitor_script, 'w', encoding='utf-8') as f:
            f.write(monitor_code)
        
        # Create batch file to launch in new window
        # IMPORTANT: All paths must be quoted to handle special characters like parentheses
        batch_content = f'''@echo off
title Training Monitor - {run_id}
cd /d "{run_dir}"
echo ============================================
echo Training Monitor
echo Run ID: {run_id}
echo ============================================
echo.
"{python_exe}" "{monitor_script}"
if errorlevel 1 (
    echo.
    echo [ERR] Monitor exited with error
    pause
)
'''
        
        batch_path = os.path.join(run_dir, 'run_training.bat')
        with open(batch_path, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        # Launch new CMD window with the batch file
        # Use call to properly handle paths with special characters
        # The title is the first quoted string after 'start'
        cmd_window = f'start "Training Monitor - {run_id}" cmd /c ""{batch_path}""'
        
        subprocess.Popen(cmd_window, shell=True)
        
        print(f"[OK] Training started in new window")
        print(f"[INFO] Log file: {log_path}")
        print(f"[INFO] Run dir: {run_dir}")
        print(f"[INFO] A new CMD window should have opened for monitoring")
        print(f"[INFO] Auto-summary will be generated when training completes")
        print(f"[HINT] Use 'python controller.py logs' to view logs here")
    
    def _start_direct_mode(self, run_dir: str, run_id: str, log_path: str, wrapper_path: str):
        """Start training in current process"""
        wrapper_dir = os.path.dirname(wrapper_path)
        
        # Detect conda environment
        conda_env = self.config.get('conda_env', 'pytorch')
        
        # Find conda python
        conda_python = self._find_conda_python(conda_env)
        if not conda_python:
            print(f"[WARN] Could not find conda env '{conda_env}', using system Python")
            conda_python = sys.executable
        else:
            print(f"[INFO] Using Python: {conda_python}")
        
        with open(log_path, 'w') as f:
            process = subprocess.Popen(
                [conda_python, wrapper_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=wrapper_dir
            )
        
        # Save PID
        pid_file = os.path.join(run_dir, 'training.pid')
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"[OK] Background started (PID: {process.pid})")
        print(f"[INFO] Log file: {log_path}")
        print(f"[INFO] Run 'python controller.py stop' to stop")
    
    def stop(self):
        """Stop training"""
        if not self._is_running():
            print("[WARN] No training is running")
            return
        
        try:
            # Get the latest run's PID
            runs = sorted(os.listdir(self.runs_dir))
            if not runs:
                print("[WARN] No runs found")
                return
            
            latest_run = runs[-1]
            pid_file = os.path.join(self.runs_dir, latest_run, 'training.pid')
            
            with open(pid_file, 'r') as f:
                pid_info = f.read().strip()
            
            # Handle window mode
            if pid_info.startswith('window:'):
                run_id = pid_info.split(':')[1]
                print(f"[INFO] Stopping window mode training (Run: {run_id})")
                # Find and kill python processes related to this run
                subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 
                              f'WINDOWTITLE eq Training Monitor - {run_id}*'], 
                             capture_output=True)
            else:
                pid = int(pid_info)
                
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/F', '/PID', str(pid)])
                else:
                    os.kill(pid, signal.SIGTERM)
            
            print(f"[OK] Training stopped")
        
        except Exception as e:
            print(f"[ERR] Stop failed: {e}")
    
    def status(self):
        """Check training status"""
        print(f"\n{'='*60}")
        print("TRAINING STATUS")
        print(f"{'='*60}")
        
        if self._is_running():
            # Get latest run info
            runs = sorted(os.listdir(self.runs_dir))
            if runs:
                latest_run = runs[-1]
                snapshot_path = os.path.join(self.runs_dir, latest_run, 'config_snapshot.json')
                if os.path.exists(snapshot_path):
                    with open(snapshot_path, 'r') as f:
                        snapshot = json.load(f)
                    print(f"\n[STATUS] Running")
                    print(f"   Run ID: {snapshot.get('run_id', latest_run)}")
                    print(f"   Log: {snapshot.get('log_file', 'N/A')}")
                    start = datetime.fromisoformat(snapshot.get('start_time'))
                    elapsed = (datetime.now() - start).total_seconds()
                    print(f"   Elapsed: {elapsed/60:.1f} minutes")
                else:
                    print("\n[STATUS] Running")
        else:
            print("\n[STATUS] Not running")
        
        # Show recent runs
        runs = sorted(os.listdir(self.runs_dir))[-5:]  # Last 5 runs
        if runs:
            print(f"\nRecent runs:")
            for run in runs:
                snapshot_path = os.path.join(self.runs_dir, run, 'config_snapshot.json')
                if os.path.exists(snapshot_path):
                    with open(snapshot_path, 'r') as f:
                        snapshot = json.load(f)
                    start = snapshot.get('start_time', 'N/A')[:16]
                    print(f"  - {run} (started: {start})")
        
        print(f"\n{'='*60}")
    
    def logs(self, run_id: str = None, n: int = 50):
        """View training logs"""
        # Determine which log file to read
        if run_id:
            log_path = os.path.join(self.runs_dir, run_id, 'train.log')
        else:
            # Use most recent run
            runs = sorted(os.listdir(self.runs_dir))
            if runs:
                log_path = os.path.join(self.runs_dir, runs[-1], 'train.log')
            else:
                print("[WARN] No training runs found")
                return
        
        if not os.path.exists(log_path):
            print(f"[WARN] Log file not found: {log_path}")
            return
        
        print(f"\n{'='*60}")
        print(f"LOG: {log_path}")
        print(f"{'='*60}\n")
        
        try:
            # Read last N lines
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines[-n:]:
                    print(line.rstrip())
        except Exception as e:
            print(f"[ERR] Failed to read log: {e}")
        
        print(f"\n{'='*60}")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run and manage training')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start training')
    start_parser.add_argument('--no-window', action='store_true', 
                             help='Run in current terminal without opening new window')
    start_parser.add_argument('--silent', action='store_true',
                             help='Run silently in background (no window, no output)')
    start_parser.add_argument('--override', nargs='+', 
                             help='Override parameters (key=value)')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop training')
    
    # Status command
    subparsers.add_parser('status', help='Check training status')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View training logs')
    logs_parser.add_argument('--run', help='Run ID to view')
    logs_parser.add_argument('-n', type=int, default=50, help='Number of lines')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    runner = TrainingRunner()
    
    if args.command == 'start':
        # Default: always open window unless --no-window or --silent is specified
        window_mode = not args.no_window and not args.silent
        overrides = {}
        if args.override:
            for item in args.override:
                if '=' in item:
                    k, v = item.split('=', 1)
                    overrides[k] = v
        runner.start(window_mode=window_mode, overrides=overrides, silent_mode=args.silent)
    
    elif args.command == 'stop':
        runner.stop()
    
    elif args.command == 'status':
        runner.status()
    
    elif args.command == 'logs':
        runner.logs(args.run, args.n)


if __name__ == '__main__':
    main()
