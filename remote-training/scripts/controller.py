#!/usr/bin/env python3
"""
Remote Training Controller - Unified control entry for ML training projects

Usage:
    python controller.py init <project_path>     # Initialize project
    python controller.py start                   # Start training
    python controller.py stop                    # Stop training
    python controller.py status                  # Check status
    python controller.py show                    # Show configuration
    python controller.py update <param> <value>  # Update parameter
    python controller.py history                 # View change history
    python controller.py logs                    # View training logs
"""

import argparse
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from init_project import init_project
from run_training import TrainingRunner
from config_manager import ConfigManager


def main():
    parser = argparse.ArgumentParser(
        description="Remote Training Controller - Universal ML training management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize a new project
  python controller.py init /path/to/project
  
  # Preview analysis without creating files
  python controller.py init /path/to/project --analyze-only
  
  # Start training (opens new window for visibility) - DEFAULT
  python controller.py start
  
  # Start in current terminal (no new window)
  python controller.py start --no-window
  
  # Start silently in background (no window, no output)
  python controller.py start --silent
  
  # Start with parameter overrides
  python controller.py start --override epochs=20 lr=0.001
  
  # Update configuration
  python controller.py update epochs 20
  python controller.py update-batch lr=0.001 batch_size=64
  
  # View logs and status
  python controller.py logs
  python controller.py logs -n 100
  python controller.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize training project')
    init_parser.add_argument('project_path', help='Path to training project')
    init_parser.add_argument('--force', action='store_true', 
                            help='Force re-initialization')
    init_parser.add_argument('--analyze-only', action='store_true',
                            help='Only analyze, do not create files')
    init_parser.add_argument('--no-llm', action='store_true',
                            help='Use regex analysis instead of LLM')
    
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
    
    # Show command
    subparsers.add_parser('show', help='Show current configuration')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update a parameter')
    update_parser.add_argument('param', help='Parameter name')
    update_parser.add_argument('value', help='New value')
    update_parser.add_argument('--note', '-n', help='Change note')
    
    # Update-batch command
    batch_parser = subparsers.add_parser('update-batch', help='Update multiple parameters')
    batch_parser.add_argument('params', nargs='+', help='key=value pairs')
    
    # History command
    hist_parser = subparsers.add_parser('history', help='View change history')
    hist_parser.add_argument('-n', type=int, default=10, help='Number of entries')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare history entries')
    compare_parser.add_argument('e1', type=int, nargs='?', default=-2)
    compare_parser.add_argument('e2', type=int, nargs='?', default=-1)
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback configuration')
    rollback_parser.add_argument('target', nargs='?', default='initial',
                                help='Target: "initial" or entry number')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View training logs')
    logs_parser.add_argument('--run', help='Run ID to view')
    logs_parser.add_argument('-n', type=int, default=50, help='Number of lines')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate/view reports')
    report_parser.add_argument('--run', help='Specific run ID')
    report_parser.add_argument('--force', action='store_true',
                              help='Force regenerate report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Route to appropriate handler
    try:
        if args.command == 'init':
            success = init_project(
                args.project_path,
                force=args.force,
                analyze_only=args.analyze_only,
                use_llm=not args.no_llm
            )
            sys.exit(0 if success else 1)
        
        elif args.command in ['start', 'stop', 'status', 'logs']:
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
                        else:
                            print(f"[WARN] Invalid override format: {item}")
                runner.start(window_mode=window_mode, overrides=overrides, silent_mode=args.silent)
            elif args.command == 'stop':
                runner.stop()
            elif args.command == 'status':
                runner.status()
            elif args.command == 'logs':
                runner.logs(args.run, args.n)
        
        elif args.command in ['show', 'update', 'update-batch', 'history', 'compare', 'rollback']:
            manager = ConfigManager()
            if args.command == 'show':
                manager.show()
            elif args.command == 'update':
                manager.update(args.param, args.value, args.note or "")
            elif args.command == 'update-batch':
                updates = {}
                for item in args.params:
                    if '=' in item:
                        k, v = item.split('=', 1)
                        updates[k] = v
                    else:
                        print(f"[WARN] Ignoring invalid format: {item}")
                manager.update_batch(updates)
            elif args.command == 'history':
                manager.history(args.n)
            elif args.command == 'compare':
                manager.compare(args.e1, args.e2)
            elif args.command == 'rollback':
                manager.rollback(args.target)
        
        elif args.command == 'report':
            from auto_reporter import generate_summary
            
            # Determine run directory
            if args.run:
                run_dir = os.path.join('remote_training', 'runs', args.run)
            else:
                # Find latest run
                remote_dir = 'remote_training'
                if not os.path.exists(remote_dir):
                    print("[ERR] remote_training directory not found")
                    sys.exit(1)
                runs_dir = os.path.join(remote_dir, 'runs')
                if not os.path.exists(runs_dir):
                    print("[ERR] No runs directory found")
                    sys.exit(1)
                runs = sorted(os.listdir(runs_dir))
                if not runs:
                    print("[ERR] No runs found")
                    sys.exit(1)
                run_dir = os.path.join(runs_dir, runs[-1])
            
            generate_summary(run_dir, force=args.force)
    
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"[ERR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
