#!/usr/bin/env python3
"""Regenerate wrapper with statistics capture"""
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from init_project import generate_smart_wrapper

# Load config
project_path = r"D:\PycharmProjects\LLM-Test\0319GatedTransformer(14)_backup_20260226_235538"
config_path = os.path.join(project_path, 'remote_training', 'config.json')

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Generate new wrapper
wrapper = generate_smart_wrapper(config['project_path'], config['main_script'], config)

# Write wrapper
wrapper_path = os.path.join(project_path, 'remote_training', 'train_wrapper.py')
with open(wrapper_path, 'w', encoding='utf-8') as f:
    f.write(wrapper)

print(f'[OK] Wrapper regenerated: {wrapper_path}')
print(f'[INFO] Length: {len(wrapper)} characters')
print(f'[INFO] Has statistics capture: {"stats = {" in wrapper}')
