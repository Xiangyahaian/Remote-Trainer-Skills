#!/usr/bin/env python3
"""
Configuration Manager - Universal parameter management
Supports: show, update, history, compare, rollback for all parameter types
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional, List


class ConfigManager:
    """Universal configuration manager for ML training projects"""
    
    def __init__(self):
        self.remote_dir = self._find_remote_dir()
        if not self.remote_dir:
            print("[ERR] remote_training directory not found")
            print("       Please run: python controller.py init <project_path>")
            sys.exit(1)
        
        self.config_path = os.path.join(self.remote_dir, 'config.json')
        self.history_path = os.path.join(self.remote_dir, 'params_history.json')
        self.config = self._load_config()
        self.history = self._load_history()
    
    def _find_remote_dir(self) -> Optional[str]:
        """Find remote_training directory in current or parent directories"""
        current = os.getcwd()
        for _ in range(5):  # Check up to 5 levels up
            remote = os.path.join(current, 'remote_training')
            if os.path.exists(remote):
                return remote
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return None
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_config(self):
        """Save configuration to file"""
        self.config['updated_at'] = datetime.now().isoformat()
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def _load_history(self) -> Dict:
        """Load parameter history"""
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'entries': []}
    
    def _save_history(self):
        """Save parameter history"""
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def _add_history_entry(self, action: str, old_values: Dict, new_values: Dict, note: str = ""):
        """Add entry to history"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'old_values': old_values,
            'new_values': new_values,
            'note': note
        }
        self.history['entries'].append(entry)
        self._save_history()
    
    def _validate_param_value(self, param_name: str, value: Any) -> Any:
        """Validate and convert parameter value based on its type"""
        if param_name not in self.config['parameters']:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        param_info = self.config['parameters'][param_name]
        param_type = param_info.get('type', 'str')
        
        try:
            if param_type == 'int':
                return int(value)
            elif param_type == 'float':
                return float(value)
            elif param_type == 'bool':
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            else:
                return str(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value '{value}' for {param_type} parameter '{param_name}'")
    
    def show(self):
        """Display current configuration"""
        print(f"\n{'='*60}")
        print("CURRENT CONFIGURATION")
        print(f"{'='*60}")
        print(f"\nProject: {self.config.get('project_path', 'N/A')}")
        print(f"Main Script: {self.config.get('main_script', 'N/A')}")
        print(f"Last Updated: {self.config.get('updated_at', 'N/A')}")
        
        print(f"\n{'='*60}")
        print("PARAMETERS")
        print(f"{'='*60}")
        
        params = self.config.get('parameters', {})
        current = self.config.get('current_values', {})
        
        # Group by source
        by_source = {}
        for name, info in params.items():
            source = info.get('source', 'unknown')
            by_source.setdefault(source, []).append((name, info, current.get(name)))
        
        for source in ['argparse', 'env_var', 'hardcoded', 'config_file']:
            if source in by_source:
                print(f"\n[{source.upper()}]")
                print("-" * 40)
                for name, info, val in sorted(by_source[source]):
                    ptype = info.get('type', 'str')
                    default = info.get('default')
                    print(f"  {name:<20} {str(val):<15} ({ptype})")
                    if val != default:
                        print(f"                       (default: {default})")
        
        print(f"\n{'='*60}")
    
    def update(self, param_name: str, value: Any, note: str = ""):
        """Update a single parameter"""
        # Validate
        converted_value = self._validate_param_value(param_name, value)
        
        old_values = self.config['current_values'].copy()
        old_value = old_values.get(param_name)
        
        # Update
        self.config['current_values'][param_name] = converted_value
        self._save_config()
        
        # History
        change_note = note or f"Changed {param_name}: {old_value} -> {converted_value}"
        self._add_history_entry('update', old_values, self.config['current_values'].copy(), change_note)
        
        print(f"[OK] Updated {param_name}: {old_value} -> {converted_value}")
        
        # Show parameter source for user info
        source = self.config['parameters'][param_name].get('source', 'unknown')
        print(f"     Source: {source}")
    
    def update_batch(self, updates: Dict[str, Any], note: str = ""):
        """Update multiple parameters at once"""
        old_values = self.config['current_values'].copy()
        changes = []
        
        for param_name, value in updates.items():
            try:
                converted_value = self._validate_param_value(param_name, value)
                old_value = self.config['current_values'].get(param_name)
                self.config['current_values'][param_name] = converted_value
                changes.append(f"{param_name}: {old_value} -> {converted_value}")
            except ValueError as e:
                print(f"[WARN] Skipping {param_name}: {e}")
        
        if changes:
            self._save_config()
            change_note = note or "; ".join(changes)
            self._add_history_entry('update_batch', old_values, self.config['current_values'].copy(), change_note)
            print(f"[OK] Updated {len(changes)} parameter(s)")
            for c in changes:
                print(f"     {c}")
        else:
            print("[WARN] No parameters were updated")
    
    def history(self, n: int = 10):
        """Show parameter change history"""
        entries = self.history.get('entries', [])
        
        if not entries:
            print("[INFO] No history entries found")
            return
        
        print(f"\n{'='*60}")
        print(f"PARAMETER HISTORY (last {min(n, len(entries))} entries)")
        print(f"{'='*60}\n")
        
        for entry in entries[-n:]:
            ts = entry.get('timestamp', 'N/A')[:19]
            action = entry.get('action', 'unknown')
            note = entry.get('note', '')
            
            print(f"[{ts}] {action.upper()}")
            if note:
                print(f"  {note}")
            print()
    
    def compare(self, entry1_idx: int = -2, entry2_idx: int = -1):
        """Compare two history entries"""
        entries = self.history.get('entries', [])
        
        if len(entries) < 2:
            print("[WARN] Not enough history entries to compare")
            return
        
        try:
            e1 = entries[entry1_idx]
            e2 = entries[entry2_idx]
        except IndexError:
            print("[ERR] Invalid entry indices")
            return
        
        v1 = e1.get('new_values', {})
        v2 = e2.get('new_values', {})
        
        print(f"\n{'='*60}")
        print(f"COMPARISON: Entry {entry1_idx} vs Entry {entry2_idx}")
        print(f"{'='*60}\n")
        
        all_params = set(v1.keys()) | set(v2.keys())
        
        print(f"{'Parameter':<20} {'Before':<15} {'After':<15}")
        print("-" * 50)
        
        for param in sorted(all_params):
            before = v1.get(param, 'N/A')
            after = v2.get(param, 'N/A')
            marker = "  *" if before != after else ""
            print(f"{param:<20} {str(before):<15} {str(after):<15}{marker}")
        
        print("\n  * = changed")
    
    def rollback(self, target: str = "initial"):
        """Rollback to a previous configuration"""
        entries = self.history.get('entries', [])
        
        if not entries:
            print("[ERR] No history entries found")
            return
        
        if target == "initial":
            target_entry = entries[0]
        elif target.isdigit():
            idx = int(target)
            if 0 <= idx < len(entries):
                target_entry = entries[idx]
            else:
                print(f"[ERR] Invalid entry index: {idx}")
                return
        else:
            print(f"[ERR] Invalid rollback target: {target}")
            print("       Use 'initial' or an entry number")
            return
        
        old_values = self.config['current_values'].copy()
        new_values = target_entry.get('new_values', {})
        
        self.config['current_values'] = new_values.copy()
        self._save_config()
        
        self._add_history_entry('rollback', old_values, new_values.copy(), 
                               f"Rolled back to {target}")
        
        print(f"[OK] Rolled back to {target}")
        print(f"     Timestamp: {target_entry.get('timestamp', 'N/A')}")


def main():
    """CLI entry point for config manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage training configuration')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Show command
    subparsers.add_parser('show', help='Show current configuration')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update parameter(s)')
    update_parser.add_argument('--param', '-p', required=True, help='Parameter name')
    update_parser.add_argument('--value', '-v', required=True, help='New value')
    update_parser.add_argument('--note', '-n', help='Change note')
    
    # Update batch command
    batch_parser = subparsers.add_parser('update-batch', help='Update multiple parameters')
    batch_parser.add_argument('params', nargs='+', help='key=value pairs')
    
    # History command
    hist_parser = subparsers.add_parser('history', help='Show change history')
    hist_parser.add_argument('-n', type=int, default=10, help='Number of entries')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare history entries')
    compare_parser.add_argument('e1', type=int, nargs='?', default=-2, help='First entry')
    compare_parser.add_argument('e2', type=int, nargs='?', default=-1, help='Second entry')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback configuration')
    rollback_parser.add_argument('target', nargs='?', default='initial', 
                                help='Target: "initial" or entry number')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
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


if __name__ == '__main__':
    main()
