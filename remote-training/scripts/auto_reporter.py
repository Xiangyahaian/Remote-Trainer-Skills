#!/usr/bin/env python3
"""
Auto Reporter - Generate comprehensive training summary reports
Parses training logs, extracts all metrics, generates detailed Markdown reports
"""

import os
import sys
import re
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path: str) -> dict:
    """Parse training log file and extract all metrics"""
    metrics = {
        'epochs': [],
        'epoch_details': defaultdict(dict),  # epoch -> {test_acc, cross_acc, loss, train_acc, time, lr, val_loss}
        'test_accuracies': [],
        'cross_accuracies': [],
        'train_accuracies': [],
        'losses': [],
        'val_losses': [],
        'learning_rates': [],
        'errors': [],
        'warnings': [],
        'raw_lines': [],
        'summary_stats': {},
        'completion_status': 'unknown',
        'wrapper_stats': {}  # Additional stats from wrapper
    }
    
    # Try to load wrapper statistics
    run_dir = os.path.dirname(log_path)
    wrapper_stats_path = os.path.join(run_dir, 'wrapper_stats.json')
    if os.path.exists(wrapper_stats_path):
        try:
            with open(wrapper_stats_path, 'r', encoding='utf-8') as f:
                metrics['wrapper_stats'] = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load wrapper stats: {e}")
    
    if not os.path.exists(log_path):
        return metrics
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
            metrics['raw_lines'] = lines
    except Exception as e:
        metrics['errors'].append(f"Failed to read log: {str(e)}")
        return metrics
    
    # Enhanced patterns for different metric formats
    patterns = {
        'epoch_start': re.compile(r'(?:Epoch|epoch)\s*[\[\s:]?(\d+)[/\s]?(\d*)', re.IGNORECASE),
        # Test accuracy: handles "Accuracy on test:95.83", "test accuracy: 95.5", "test: 95.5%"
        'test_acc': re.compile(r'(?:Accuracy on test|test.*?acc|test.*?:)\s*[:\s]+(\d+\.?\d*)\s*%?', re.IGNORECASE),
        # Cross/Val accuracy: handles "Accuracy on cross:95.79", "cross accuracy: 95.5", "val acc: 95.5"
        'cross_acc': re.compile(r'(?:Accuracy on cross|cross.*?acc|val.*?acc|validation.*?acc|cross.*?:|val.*?:)\s*[:\s]+(\d+\.?\d*)\s*%?', re.IGNORECASE),
        # Train accuracy: handles "train accuracy: 95.5", "training acc: 95.5", "train: 95.5%"
        'train_acc': re.compile(r'(?:train.*?acc|training.*?acc|train.*?:)\s*[:\s]+(\d+\.?\d*)\s*%?', re.IGNORECASE),
        # Loss: handles "loss: 0.234", "train_loss: 0.234", "Loss=0.234"
        'loss': re.compile(r'(?:loss|train_loss)\s*[:=\s]+(\d+\.?\d*)', re.IGNORECASE),
        # Validation Loss: handles "val_loss: 0.234", "validation_loss=0.234", "test_loss: 0.234"
        'val_loss': re.compile(r'(?:val_loss|validation_loss|test_loss)\s*[:=\s]+(\d+\.?\d*)', re.IGNORECASE),
        # Learning rate: handles "lr: 0.001", "learning_rate=1e-4", "LR: 0.0002"
        'lr': re.compile(r'(?:lr|learning_rate)\s*[:=\s]+(\d+\.?\d*(?:e-?\d+)?)', re.IGNORECASE),
        # Time per epoch: handles "time: 45.2s", "duration: 10.5 sec", "elapsed: 5.2"
        'time': re.compile(r'(?:time|duration|elapsed)\s*[:\s]+(\d+\.?\d*)\s*(?:s|sec|seconds)?', re.IGNORECASE),
        # Total time at end: handles "共耗时: 2.84", "total time: 120.5s"
        'total_time': re.compile(r'(?:共耗时|total time|total_time)\s*[:\s]+(\d+\.?\d*)', re.IGNORECASE),
        'completed': re.compile(r'(?:training\s+complete|finished|done|completed|best model saved|final accuracy)', re.IGNORECASE),
    }
    
    current_epoch = None
    last_test_acc = None
    last_cross_acc = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # Extract epoch number
        epoch_match = patterns['epoch_start'].search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in metrics['epochs']:
                metrics['epochs'].append(current_epoch)
                metrics['epoch_details'][current_epoch]['line_num'] = i + 1
        
        # Extract all metrics for current epoch
        if current_epoch is not None:
            # Handle lines with both test and cross accuracy
            # Format: "Epoch:1 Accuracy on test:90.96, Accuracy on cross:88.36"
            if 'accuracy on test' in line.lower() and 'accuracy on cross' in line.lower():
                # Extract both values from the same line
                test_match = re.search(r'accuracy on test[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
                cross_match = re.search(r'accuracy on cross[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
                
                if test_match:
                    acc = float(test_match.group(1))
                    if acc <= 1.0:
                        acc *= 100
                    metrics['test_accuracies'].append((current_epoch, acc))
                    metrics['epoch_details'][current_epoch]['test_acc'] = acc
                    last_test_acc = acc
                
                if cross_match:
                    acc = float(cross_match.group(1))
                    if acc <= 1.0:
                        acc *= 100
                    metrics['cross_accuracies'].append((current_epoch, acc))
                    metrics['epoch_details'][current_epoch]['cross_acc'] = acc
                    last_cross_acc = acc
            else:
                # Handle separate lines or combined format
                # For combined format like: "Epoch [1/32] ... | Test Acc: 89.91% | Cross Acc: 85.48%"
                # Extract all accuracy values from the line
                
                # Use specific patterns for combined format: "Test Acc: 89.91%"
                test_acc_match = re.search(r'Test Acc[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
                if test_acc_match:
                    acc = float(test_acc_match.group(1))
                    if acc <= 1.0:
                        acc *= 100
                    metrics['test_accuracies'].append((current_epoch, acc))
                    metrics['epoch_details'][current_epoch]['test_acc'] = acc
                    last_test_acc = acc
                else:
                    # Fallback to general test accuracy pattern
                    test_match = patterns['test_acc'].search(line)
                    if test_match and 'cross' not in line.lower():
                        acc = float(test_match.group(1))
                        if acc <= 1.0:
                            acc *= 100
                        metrics['test_accuracies'].append((current_epoch, acc))
                        metrics['epoch_details'][current_epoch]['test_acc'] = acc
                        last_test_acc = acc
                
                # Cross/Val accuracy - try specific pattern first
                cross_acc_match = re.search(r'Cross Acc[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
                if cross_acc_match:
                    acc = float(cross_acc_match.group(1))
                    if acc <= 1.0:
                        acc *= 100
                    if current_epoch not in metrics['epoch_details'] or \
                       'cross_acc' not in metrics['epoch_details'][current_epoch]:
                        metrics['cross_accuracies'].append((current_epoch, acc))
                        metrics['epoch_details'][current_epoch]['cross_acc'] = acc
                        last_cross_acc = acc
                else:
                    # Fallback to general cross accuracy pattern
                    cross_match = patterns['cross_acc'].search(line)
                    if cross_match and ('cross' in line.lower() or 'val' in line.lower()):
                        acc = float(cross_match.group(1))
                        if acc <= 1.0:
                            acc *= 100
                        if current_epoch not in metrics['epoch_details'] or \
                           'cross_acc' not in metrics['epoch_details'][current_epoch]:
                            metrics['cross_accuracies'].append((current_epoch, acc))
                            metrics['epoch_details'][current_epoch]['cross_acc'] = acc
                            last_cross_acc = acc
            
            # Train accuracy
            train_match = patterns['train_acc'].search(line)
            if train_match:
                acc = float(train_match.group(1))
                if acc <= 1.0:
                    acc *= 100
                metrics['train_accuracies'].append((current_epoch, acc))
                metrics['epoch_details'][current_epoch]['train_acc'] = acc
            
            # Loss
            loss_match = patterns['loss'].search(line)
            if loss_match:
                loss = float(loss_match.group(1))
                metrics['losses'].append((current_epoch, loss))
                metrics['epoch_details'][current_epoch]['loss'] = loss
            
            # Validation Loss
            val_loss_match = patterns['val_loss'].search(line)
            if val_loss_match:
                val_loss = float(val_loss_match.group(1))
                metrics['val_losses'].append((current_epoch, val_loss))
                metrics['epoch_details'][current_epoch]['val_loss'] = val_loss
            
            # Learning rate
            lr_match = patterns['lr'].search(line)
            if lr_match:
                lr = float(lr_match.group(1))
                metrics['learning_rates'].append((current_epoch, lr))
                metrics['epoch_details'][current_epoch]['lr'] = lr
            
            # Time
            time_match = patterns['time'].search(line)
            if time_match:
                time_val = float(time_match.group(1))
                metrics['epoch_details'][current_epoch]['time'] = time_val
        
        # Check for completion
        if patterns['completed'].search(line):
            metrics['completion_status'] = 'completed'
        
        # Extract total time at end of log
        total_time_match = patterns.get('total_time', re.compile(r'never')).search(line)
        if total_time_match:
            metrics['total_time'] = float(total_time_match.group(1))
    
    # Extract train accuracy from array at end of log (format: [91.68, 90.38, 93.83...])
    # Look for arrays that appear after all epochs
    train_acc_array_pattern = re.compile(r'\[([\d.,\s]+)\]')
    array_matches = []
    for i, line in enumerate(lines):
        match = train_acc_array_pattern.search(line)
        if match:
            array_matches.append((i, match.group(1)))
    
    # If we found arrays and have epoch data, the first array might be train acc
    if array_matches and metrics['epochs']:
        # Use the array that appears after epoch data (usually the first one)
        for line_num, array_content in array_matches:
            try:
                values = [float(x.strip()) for x in array_content.split(',')]
                # Check if array length matches epoch count
                if len(values) == len(metrics['epochs']):
                    for i, epoch in enumerate(sorted(metrics['epochs'])):
                        acc = values[i]
                        if acc <= 1.0:
                            acc *= 100
                        metrics['train_accuracies'].append((epoch, acc))
                        metrics['epoch_details'][epoch]['train_acc'] = acc
                    break
            except:
                continue
        
        # Check for errors and warnings
        if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed', 'failure']):
            if not line.startswith('[WRAPPER]'):
                metrics['errors'].append({
                    'line': i + 1,
                    'content': line_stripped[:200]  # Limit length
                })
        
        if any(keyword in line.lower() for keyword in ['warning', 'warn', 'deprecated']):
            metrics['warnings'].append({
                'line': i + 1,
                'content': line_stripped[:200]
            })
    
    # Calculate summary statistics
    if metrics['test_accuracies']:
        accs = [acc for _, acc in metrics['test_accuracies']]
        metrics['summary_stats']['test_acc_max'] = max(accs)
        metrics['summary_stats']['test_acc_min'] = min(accs)
        metrics['summary_stats']['test_acc_avg'] = sum(accs) / len(accs)
        metrics['summary_stats']['test_acc_final'] = accs[-1]
    
    if metrics['cross_accuracies']:
        accs = [acc for _, acc in metrics['cross_accuracies']]
        metrics['summary_stats']['cross_acc_max'] = max(accs)
        metrics['summary_stats']['cross_acc_final'] = accs[-1]
    
    if metrics['train_accuracies']:
        accs = [acc for _, acc in metrics['train_accuracies']]
        metrics['summary_stats']['train_acc_max'] = max(accs)
        metrics['summary_stats']['train_acc_final'] = accs[-1]
    
    if metrics['losses']:
        losses = [loss for _, loss in metrics['losses']]
        metrics['summary_stats']['loss_min'] = min(losses)
        metrics['summary_stats']['loss_max'] = max(losses)
        metrics['summary_stats']['loss_final'] = losses[-1]
    
    if metrics['val_losses']:
        val_losses = [vl for _, vl in metrics['val_losses']]
        metrics['summary_stats']['val_loss_min'] = min(val_losses)
        metrics['summary_stats']['val_loss_final'] = val_losses[-1]
    
    # Merge wrapper statistics (Loss, LR, Time from wrapper monitoring)
    wrapper_stats = metrics.get('wrapper_stats', {})
    wrapper_epochs = wrapper_stats.get('epochs', {})
    
    # Try to load config to get LR and scheduler info
    config_lr = None
    config_scheduler = None
    try:
        config_path = os.path.join(os.path.dirname(log_path), '..', '..', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                current_values = config.get('current_values', {})
                config_lr = current_values.get('lr')
                config_scheduler = current_values.get('scheduler', '')
    except:
        pass
    
    for epoch_str, epoch_data in wrapper_epochs.items():
        try:
            epoch = int(epoch_str)
            if epoch not in metrics['epochs']:
                metrics['epochs'].append(epoch)
            
            # Add Loss from wrapper if not already extracted from log
            if 'loss' in epoch_data and 'loss' not in metrics['epoch_details'][epoch]:
                loss_val = epoch_data['loss']
                metrics['epoch_details'][epoch]['loss'] = loss_val
                metrics['losses'].append((epoch, loss_val))
            
            # Add Val Loss from wrapper
            if 'val_loss' in epoch_data and 'val_loss' not in metrics['epoch_details'][epoch]:
                val_loss_val = epoch_data['val_loss']
                metrics['epoch_details'][epoch]['val_loss'] = val_loss_val
                metrics['val_losses'].append((epoch, val_loss_val))
            
            # Add LR from wrapper
            if 'lr' in epoch_data and 'lr' not in metrics['epoch_details'][epoch]:
                lr_val = epoch_data['lr']
                metrics['epoch_details'][epoch]['lr'] = lr_val
                metrics['learning_rates'].append((epoch, lr_val))
            
            # Add Time from wrapper
            if 'time' in epoch_data and 'time' not in metrics['epoch_details'][epoch]:
                metrics['epoch_details'][epoch]['time'] = epoch_data['time']
        except:
            pass
    
    # Update summary stats with wrapper data if needed
    if wrapper_epochs and not metrics['losses']:
        for epoch_str, epoch_data in wrapper_epochs.items():
            if 'loss' in epoch_data:
                try:
                    epoch = int(epoch_str)
                    metrics['losses'].append((epoch, epoch_data['loss']))
                except:
                    pass
    
    if wrapper_epochs and not metrics['learning_rates']:
        for epoch_str, epoch_data in wrapper_epochs.items():
            if 'lr' in epoch_data:
                try:
                    epoch = int(epoch_str)
                    metrics['learning_rates'].append((epoch, epoch_data['lr']))
                except:
                    pass
    
    # Re-calculate summary stats with merged data
    if metrics['losses']:
        losses = [loss for _, loss in metrics['losses']]
        metrics['summary_stats']['loss_min'] = min(losses)
        metrics['summary_stats']['loss_max'] = max(losses)
        metrics['summary_stats']['loss_final'] = losses[-1]
    
    if metrics['val_losses']:
        val_losses = [vl for _, vl in metrics['val_losses']]
        metrics['summary_stats']['val_loss_min'] = min(val_losses)
        metrics['summary_stats']['val_loss_final'] = val_losses[-1]
    
    # Calculate per-epoch LR from config if available
    if config_lr and metrics['epochs']:
        try:
            initial_lr = float(config_lr)
            step_size = 5  # Default StepLR step_size
            gamma = 0.5    # Default StepLR gamma
            
            # Parse scheduler config if available
            if config_scheduler and 'StepLR' in str(config_scheduler):
                # Try to extract step_size and gamma from scheduler string
                # Format: StepLR(optimizer, step_size=5, gamma=0.5)
                step_match = re.search(r'step_size\s*=\s*(\d+)', str(config_scheduler))
                gamma_match = re.search(r'gamma\s*=\s*(0?\.\d+)', str(config_scheduler))
                if step_match:
                    step_size = int(step_match.group(1))
                if gamma_match:
                    gamma = float(gamma_match.group(1))
            
            # Calculate LR for each epoch
            for epoch in metrics['epochs']:
                # StepLR: lr = initial_lr * gamma^(epoch // step_size)
                lr_value = initial_lr * (gamma ** ((epoch - 1) // step_size))
                if 'lr' not in metrics['epoch_details'][epoch]:
                    metrics['epoch_details'][epoch]['lr'] = lr_value
                    metrics['learning_rates'].append((epoch, lr_value))
            
            metrics['learning_rates'] = sorted(set(metrics['learning_rates']), key=lambda x: x[0])
        except:
            pass
    
    return metrics


def find_best_performance(metrics: dict) -> dict:
    """Find best performance across all epochs"""
    best = {
        'test_acc': {'value': 0, 'epoch': None},
        'cross_acc': {'value': 0, 'epoch': None},
        'train_acc': {'value': 0, 'epoch': None},
        'loss': {'value': float('inf'), 'epoch': None},
        'val_loss': {'value': float('inf'), 'epoch': None},
        'combined': {'value': 0, 'epoch': None}
    }
    
    for epoch, acc in metrics['test_accuracies']:
        if acc > best['test_acc']['value']:
            best['test_acc'] = {'value': acc, 'epoch': epoch}
    
    for epoch, acc in metrics['cross_accuracies']:
        if acc > best['cross_acc']['value']:
            best['cross_acc'] = {'value': acc, 'epoch': epoch}
    
    for epoch, acc in metrics['train_accuracies']:
        if acc > best['train_acc']['value']:
            best['train_acc'] = {'value': acc, 'epoch': epoch}
    
    for epoch, loss in metrics['losses']:
        if loss < best['loss']['value']:
            best['loss'] = {'value': loss, 'epoch': epoch}
    
    for epoch, val_loss in metrics['val_losses']:
        if val_loss < best['val_loss']['value']:
            best['val_loss'] = {'value': val_loss, 'epoch': epoch}
    
    # Find best combined (test + cross)
    combined_scores = {}
    for epoch, acc in metrics['test_accuracies']:
        combined_scores[epoch] = {'test': acc, 'cross': 0}
    for epoch, acc in metrics['cross_accuracies']:
        if epoch in combined_scores:
            combined_scores[epoch]['cross'] = acc
        else:
            combined_scores[epoch] = {'test': 0, 'cross': acc}
    
    for epoch, scores in combined_scores.items():
        combined = scores['test'] + scores['cross']
        if combined > best['combined']['value']:
            best['combined'] = {'value': combined, 'epoch': epoch}
    
    return best


def generate_ascii_chart(values, width=50, height=10, label=""):
    """Generate ASCII chart for visualizing trends"""
    if not values:
        return "No data"
    
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        max_val = min_val + 1
    
    chart = []
    chart.append(f"{label}")
    chart.append(f"Range: {min_val:.2f} - {max_val:.2f}")
    chart.append("")
    
    for row in range(height, 0, -1):
        threshold = min_val + (max_val - min_val) * (row - 1) / height
        line = f"{threshold:6.2f} |"
        for val in values:
            if val >= threshold:
                line += "*"
            else:
                line += " "
        chart.append(line)
    
    chart.append("       +" + "-" * len(values))
    chart.append(f"        {'Start':^{len(values)//2}}{'End':^{len(values)//2}}")
    
    return "\n".join(chart)


def format_epoch_table(metrics: dict) -> str:
    """Format detailed epoch-by-epoch table with all available metrics"""
    lines = []
    
    # Check which columns have data (for data source annotation)
    has_loss = any('loss' in details for details in metrics['epoch_details'].values())
    has_val_loss = any('val_loss' in details for details in metrics['epoch_details'].values())
    has_lr = any('lr' in details for details in metrics['epoch_details'].values())
    has_time = any('time' in details for details in metrics['epoch_details'].values())
    
    # Header - all possible columns
    lines.append("| Epoch | Test Acc | Cross Acc | Train Acc | Loss | Val Loss | LR | Time |")
    lines.append("|-------|----------|-----------|-----------|------|----------|-------|------|")
    
    # Sort epochs
    sorted_epochs = sorted(metrics['epoch_details'].keys())
    
    for epoch in sorted_epochs:
        details = metrics['epoch_details'][epoch]
        
        # Format each value - show "-" if not available
        test_acc = f"{details.get('test_acc', 0):.2f}%" if 'test_acc' in details else "-"
        cross_acc = f"{details.get('cross_acc', 0):.2f}%" if 'cross_acc' in details else "-"
        train_acc = f"{details.get('train_acc', 0):.2f}%" if 'train_acc' in details else "-"
        loss = f"{details.get('loss', 0):.4f}" if 'loss' in details else "-"
        val_loss = f"{details.get('val_loss', 0):.4f}" if 'val_loss' in details else "-"
        lr = f"{details.get('lr', 0):.2e}" if 'lr' in details else "-"
        time_val = f"{details.get('time', 0):.1f}s" if 'time' in details else "-"
        
        lines.append(f"| {epoch:5d} | {test_acc:>8} | {cross_acc:>9} | {train_acc:>9} | {loss:>4} | {val_loss:>8} | {lr:>5} | {time_val:>4} |")
    
    # Add legend explaining availability and sources
    lines.append("")
    lines.append("**Legend:** `-` = Metric not available")
    lines.append("")
    lines.append("**Data Sources:**")
    lines.append("- `Test/Cross/Train Acc`: Extracted from training log output")
    if has_loss or has_val_loss or has_lr or has_time:
        sources = []
        if has_loss:
            sources.append("Loss")
        if has_val_loss:
            sources.append("Val Loss")
        if has_lr:
            sources.append("LR")
        if has_time:
            sources.append("Time")
        lines.append(f"- `{', '.join(sources)}`: Captured by wrapper from script output or calculated")
    lines.append("")
    lines.append("**Detection Patterns:**")
    lines.append("- `Test Acc`: 'test accuracy: 95.5', 'Accuracy on test: 95.5%'")
    lines.append("- `Cross Acc`: 'cross accuracy', 'val accuracy: 95.5%'")
    lines.append("- `Train Acc`: 'train accuracy', 'training acc: 95.5%'")
    lines.append("- `Loss`: 'loss: 0.234', 'train_loss: 0.234', 'Loss=0.5'")
    lines.append("- `Val Loss`: 'val_loss: 0.234', 'validation_loss=0.234'")
    lines.append("- `LR`: 'lr: 0.001', 'learning_rate: 1e-4', 'LR=0.0002'")
    lines.append("- `Time`: Calculated by wrapper (per-epoch duration)")
    
    return "\n".join(lines)


def generate_training_summary(metrics: dict) -> str:
    """Generate training progress summary with statistics"""
    lines = []
    
    stats = metrics['summary_stats']
    
    if 'test_acc_max' in stats:
        lines.append("### Test Accuracy Statistics")
        lines.append(f"- **Maximum**: {stats['test_acc_max']:.2f}%")
        lines.append(f"- **Minimum**: {stats['test_acc_min']:.2f}%")
        lines.append(f"- **Average**: {stats['test_acc_avg']:.2f}%")
        lines.append(f"- **Final**: {stats['test_acc_final']:.2f}%")
        lines.append("")
        
        # Calculate improvement
        if stats['test_acc_final'] > stats['test_acc_min']:
            improvement = stats['test_acc_final'] - stats['test_acc_min']
            lines.append(f"- **Total Improvement**: +{improvement:.2f}%")
        lines.append("")
    
    if 'cross_acc_max' in stats:
        lines.append("### Cross/Validation Accuracy Statistics")
        lines.append(f"- **Maximum**: {stats['cross_acc_max']:.2f}%")
        lines.append(f"- **Final**: {stats['cross_acc_final']:.2f}%")
        lines.append("")
    
    if 'train_acc_max' in stats:
        lines.append("### Training Accuracy Statistics")
        lines.append(f"- **Maximum**: {stats['train_acc_max']:.2f}%")
        lines.append(f"- **Final**: {stats['train_acc_final']:.2f}%")
        lines.append("")
    
    if 'loss_min' in stats:
        lines.append("### Loss Statistics")
        lines.append(f"- **Minimum**: {stats['loss_min']:.4f}")
        lines.append(f"- **Maximum**: {stats['loss_max']:.4f}")
        lines.append(f"- **Final**: {stats['loss_final']:.4f}")
        lines.append("")
    
    if 'val_loss_min' in stats:
        lines.append("### Validation Loss Statistics")
        lines.append(f"- **Minimum**: {stats['val_loss_min']:.4f}")
        lines.append(f"- **Final**: {stats['val_loss_final']:.4f}")
        lines.append("")
    
    # Learning Rate Statistics
    if metrics['learning_rates']:
        lrs = [lr for _, lr in sorted(metrics['learning_rates'])]
        if lrs:
            lines.append("### Learning Rate Statistics")
            lines.append(f"- **Initial**: {lrs[0]:.2e}")
            lines.append(f"- **Final**: {lrs[-1]:.2e}")
            if len(lrs) > 1:
                lines.append(f"- **Minimum**: {min(lrs):.2e}")
                lines.append(f"- **Maximum**: {max(lrs):.2e}")
            lines.append("")
    
    # Time Statistics
    epoch_times = []
    for epoch, details in metrics['epoch_details'].items():
        if 'time' in details:
            epoch_times.append(details['time'])
    
    if epoch_times:
        lines.append("### Time Statistics")
        lines.append(f"- **Total Epochs**: {len(epoch_times)}")
        lines.append(f"- **Average Epoch Time**: {sum(epoch_times)/len(epoch_times):.2f}s")
        lines.append(f"- **Fastest Epoch**: {min(epoch_times):.2f}s")
        lines.append(f"- **Slowest Epoch**: {max(epoch_times):.2f}s")
        lines.append(f"- **Total Training Time**: {sum(epoch_times):.2f}s ({sum(epoch_times)/60:.2f} min)")
        lines.append("")
    
    return "\n".join(lines)


def generate_summary(run_dir: str, force: bool = False) -> str:
    """Generate comprehensive training summary report"""
    summary_path = os.path.join(run_dir, 'summary.md')
    
    if os.path.exists(summary_path) and not force:
        print(f"[INFO] Summary already exists: {summary_path}")
        return summary_path
    
    # Find log file
    log_path = os.path.join(run_dir, 'train.log')
    if not os.path.exists(log_path):
        print(f"[ERR] Log file not found: {log_path}")
        return None
    
    # Load snapshot for config info
    snapshot_path = os.path.join(run_dir, 'config_snapshot.json')
    config_info = {}
    if os.path.exists(snapshot_path):
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            config_info = json.load(f)
    
    # Parse metrics
    metrics = parse_log_file(log_path)
    best = find_best_performance(metrics)
    
    # Generate report metadata
    run_id = os.path.basename(run_dir)
    start_time = config_info.get('start_time', 'Unknown')
    end_time = datetime.now().isoformat()
    
    # Calculate duration
    duration_str = "Unknown"
    duration_seconds = 0
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        duration = (end_dt - start_dt).total_seconds()
        duration_seconds = duration
        if duration < 60:
            duration_str = f"{duration:.1f} seconds"
        elif duration < 3600:
            duration_str = f"{duration/60:.1f} minutes"
        else:
            duration_str = f"{duration/3600:.2f} hours"
    except:
        pass
    
    # Get configuration
    current_values = config_info.get('current_values', {})
    param_types = config_info.get('parameter_types', {})
    
    # Build comprehensive markdown report - ALL IN ENGLISH
    lines = [
        f"# Training Experiment Report",
        "",
        f"## Overview",
        f"",
        f"| Attribute | Value |",
        f"|-----------|-------|",
        f"| **Run ID** | `{run_id}` |",
        f"| **Start Time** | {start_time} |",
        f"| **End Time** | {end_time} |",
        f"| **Duration** | {duration_str} |",
        f"| **Status** | {'Completed' if metrics['completion_status'] == 'completed' else 'Incomplete/Interrupted'} |",
        f"| **Total Epochs** | {max(metrics['epochs']) if metrics['epochs'] else 'N/A'} |",
        f"",
        f"---",
        f"",
        f"## Configuration Parameters",
        f"",
        f"| Parameter | Value | Type |",
        f"|-----------|-------|------|",
    ]
    
    # Sort parameters by type for better organization
    sorted_params = sorted(current_values.items(), 
                          key=lambda x: param_types.get(x[0], 'unknown'))
    
    for param, value in sorted_params:
        ptype = param_types.get(param, 'unknown')
        type_label = {
            'argparse': 'CLI Argument',
            'env_var': 'Environment Variable',
            'hardcoded': 'Hardcoded',
            'unknown': 'Unknown'
        }.get(ptype, ptype)
        lines.append(f"| `{param}` | `{value}` | {type_label} |")
    
    lines.extend([
        "",
        "---",
        "",
        f"## Best Performance",
        "",
    ])
    
    # Best performance section
    if best['test_acc']['epoch'] is not None:
        lines.append(f"### Accuracy")
        lines.append(f"- **Best Test Accuracy**: `{best['test_acc']['value']:.2f}%` (Epoch {best['test_acc']['epoch']}) :star:")
        
        if best['cross_acc']['epoch'] is not None:
            lines.append(f"- **Best Cross/Val Accuracy**: `{best['cross_acc']['value']:.2f}%` (Epoch {best['cross_acc']['epoch']})")
        
        if best['train_acc']['epoch'] is not None:
            lines.append(f"- **Best Train Accuracy**: `{best['train_acc']['value']:.2f}%` (Epoch {best['train_acc']['epoch']})")
        
        if best['combined']['epoch'] is not None:
            lines.append(f"- **Best Combined Performance**: Epoch {best['combined']['epoch']} (Test + Cross Total: {best['combined']['value']:.2f})")
        
        lines.append("")
    
    if best['loss']['epoch'] is not None:
        lines.append(f"### Loss")
        lines.append(f"- **Lowest Loss**: `{best['loss']['value']:.4f}` (Epoch {best['loss']['epoch']}) :star:")
        lines.append("")
    
    if best['val_loss']['epoch'] is not None:
        lines.append(f"### Validation Loss")
        lines.append(f"- **Lowest Val Loss**: `{best['val_loss']['value']:.4f}` (Epoch {best['val_loss']['epoch']}) :star:")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        f"## Training Statistics",
        "",
    ])
    
    # Add detailed statistics
    lines.append(generate_training_summary(metrics))
    
    lines.extend([
        "---",
        "",
        f"## Epoch Details",
        "",
        f"> Detailed training metrics for each epoch. Only metrics detected in the training log are shown.",
        "",
    ])
    
    # Add detailed epoch table
    if metrics['epochs']:
        lines.append(format_epoch_table(metrics))
    else:
        lines.append("*No epoch data detected*")
    
    lines.append("")
    
    # Visual charts (if enough data points)
    if len(metrics['test_accuracies']) > 2:
        lines.extend([
            "---",
            "",
            f"## Trend Visualization",
            "",
            f"### Test Accuracy Trend",
            "```",
        ])
        test_accs = [acc for _, acc in sorted(metrics['test_accuracies'])]
        lines.append(generate_ascii_chart(test_accs, label="Test Accuracy"))
        lines.append("```")
        lines.append("")
    
    if len(metrics['cross_accuracies']) > 2:
        lines.extend([
            f"### Cross/Validation Accuracy Trend",
            "```",
        ])
        cross_accs = [acc for _, acc in sorted(metrics['cross_accuracies'])]
        lines.append(generate_ascii_chart(cross_accs, label="Cross Accuracy"))
        lines.append("```")
        lines.append("")
    
    if len(metrics['train_accuracies']) > 2:
        lines.extend([
            f"### Training Accuracy Trend",
            "```",
        ])
        train_accs = [acc for _, acc in sorted(metrics['train_accuracies'])]
        lines.append(generate_ascii_chart(train_accs, label="Train Accuracy"))
        lines.append("```")
        lines.append("")
    
    if len(metrics['losses']) > 2:
        lines.extend([
            f"### Loss Trend",
            "```",
        ])
        losses = [loss for _, loss in sorted(metrics['losses'])]
        lines.append(generate_ascii_chart(losses, label="Loss"))
        lines.append("```")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        f"## Raw Training Log",
        "",
        f"<details>",
        f"<summary>Click to expand full log ({len(metrics['raw_lines'])} lines)</summary>",
        "",
        "```text",
    ])
    
    # Add raw log content (first 500 lines to avoid huge files)
    max_lines = 500
    for i, line in enumerate(metrics['raw_lines'][:max_lines]):
        # Escape problematic characters for markdown
        safe_line = line.replace('```', '` ``').rstrip()
        if safe_line:
            lines.append(safe_line)
    
    if len(metrics['raw_lines']) > max_lines:
        lines.append(f"\n... ({len(metrics['raw_lines']) - max_lines} more lines, see train.log for full log)")
    
    lines.extend([
        "```",
        "",
        "</details>",
        "",
    ])
    
    # Errors and warnings section
    if metrics['errors'] or metrics['warnings']:
        lines.extend([
            "---",
            "",
            f"## Errors and Warnings",
            "",
        ])
        
        if metrics['errors']:
            lines.append(f"### Errors ({len(metrics['errors'])} total)")
            lines.append("")
            for err in metrics['errors'][:20]:  # Show first 20
                lines.append(f"- **Line {err['line']}**: `{err['content'][:100]}`")
            if len(metrics['errors']) > 20:
                lines.append(f"- ... {len(metrics['errors']) - 20} more errors")
            lines.append("")
        
        if metrics['warnings']:
            lines.append(f"### Warnings ({len(metrics['warnings'])} total)")
            lines.append("")
            for warn in metrics['warnings'][:10]:  # Show first 10
                lines.append(f"- **Line {warn['line']}**: `{warn['content'][:100]}`")
            if len(metrics['warnings']) > 10:
                lines.append(f"- ... {len(metrics['warnings']) - 10} more warnings")
            lines.append("")
    
    # File references
    lines.extend([
        "---",
        "",
        f"## Related Files",
        "",
        f"| File | Description |",
        f"|------|-------------|",
        f"| [`train.log`](train.log) | Complete training log |",
        f"| [`config_snapshot.json`](config_snapshot.json) | Configuration snapshot for this run |",
        f"| [`training.pid`](training.pid) | Training process ID |",
        f"",
        "---",
        f"",
        f"*Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  ",
        f"*Generated by remote-training skill*",
    ])
    
    # Write report
    report_content = "\n".join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"[OK] Summary generated: {summary_path}")
    print(f"[INFO] Report includes:")
    print(f"       - {len(metrics['epochs'])} epochs data")
    print(f"       - {len(metrics['raw_lines'])} log lines")
    print(f"       - {len(metrics['errors'])} errors")
    print(f"       - {len(metrics['warnings'])} warnings")
    return summary_path


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training summary reports')
    parser.add_argument('run_dir', nargs='?', help='Run directory path')
    parser.add_argument('--force', action='store_true', help='Force regenerate')
    parser.add_argument('--watch', action='store_true', help='Watch for training completion')
    parser.add_argument('--latest', action='store_true', help='Use latest run')
    
    args = parser.parse_args()
    
    # Determine run directory
    if args.latest or not args.run_dir:
        # Find latest run
        remote_dir = None
        current = os.getcwd()
        for _ in range(5):
            r = os.path.join(current, 'remote_training')
            if os.path.exists(r):
                remote_dir = r
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        
        if not remote_dir:
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
    else:
        run_dir = args.run_dir
    
    if args.watch:
        print(f"[WATCH] Waiting for training to complete: {run_dir}")
        log_path = os.path.join(run_dir, 'train.log')
        
        # Wait for log file
        while not os.path.exists(log_path):
            time.sleep(1)
        
        # Monitor for completion
        last_size = 0
        stable_count = 0
        
        while True:
            try:
                current_size = os.path.getsize(log_path)
                if current_size == last_size:
                    stable_count += 1
                    if stable_count > 30:  # 30 seconds of no changes
                        break
                else:
                    stable_count = 0
                    last_size = current_size
                    
                    # Check for completion patterns
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        if any(p in content for p in ['training complete', 'finished', 'done']):
                            time.sleep(5)
                            break
            except:
                pass
            time.sleep(1)
        
        print("[WATCH] Training appears complete, generating report...")
    
    generate_summary(run_dir, force=args.force)


if __name__ == '__main__':
    import time
    main()
