#!/usr/bin/env python3
"""
LLM Parameter Analyzer - Universal parameter extraction for ML training scripts
Supports: PyTorch, TensorFlow, JAX, and generic Python training scripts
"""

import os
import sys
import json
import re
from typing import Dict, List, Any, Tuple, Optional


# =============================================================================
# LLM Prompt Template
# =============================================================================

SYSTEM_PROMPT = """You are an expert Python code analyzer specializing in machine learning training scripts.
Your task is to analyze the provided training script and extract ALL parameters that can affect model training.

Analyze the code deeply - look at imports, function definitions, global variables, argparse setup, and main execution blocks.

Extract these parameter categories:
1. **Training Hyperparameters**: epochs, batch_size, learning_rate, weight_decay, warmup_steps, gradient_accumulation_steps
2. **Model Architecture**: hidden_dim, num_layers, num_heads, dropout_rate, d_model, vocab_size
3. **Data Processing**: sequence_length, max_seq_len, num_workers, train_split, augmentation
4. **Optimizer Settings**: optimizer_type (adam, sgd, adamw), scheduler (cosine, linear), beta1, beta2
5. **Hardware/Runtime**: device (cuda/cpu), num_gpus, fp16/bf16, seed
6. **Paths**: data_path, save_dir, checkpoint_path, log_dir
7. **Logging/Evaluation**: log_interval, eval_interval, save_interval, metric_for_best_model

For each parameter, provide:
- name: Parameter name (snake_case)
- type: Python type (int, float, str, bool, list)
- default: Default value
- source: How it's defined - "argparse", "env_var", "hardcoded", "config_file", or "dataclass"
- metadata: Additional info like arg_name (for argparse), env_var (for env), var_name (for hardcoded)
- modifiable: Always set to true (all detected parameters can be modified via wrapper)

IMPORTANT: All parameters that affect training should be marked as modifiable=true."""

USER_PROMPT_TEMPLATE = """Analyze this Python training script and extract all training parameters, and recommend the most suitable Conda environment.

Script content:
```python
{script_content}
```

Return ONLY a valid JSON object in this exact format:
{{
  "parameters": {{
    "param_name": {{
      "type": "int|float|str|bool|list",
      "default": <value>,
      "description": "What this parameter controls",
      "source": "argparse|env_var|hardcoded|config_file|dataclass",
      "arg_name": "--param-name" (if argparse),
      "env_var": "PARAM_NAME" (if env_var),
      "var_name": "PARAM_NAME" (if hardcoded),
      "modifiable": true
    }}
  }},
  "main_entrypoint": {{
    "script": "filename.py",
    "function": "main|train|None",
    "has_argparse": true|false,
    "has_config_file": true|false
  }},
  "conda_environment": {{
    "recommended": "recommended_env_name",
    "confidence": "high|medium|low",
    "reasoning": "Detailed explanation of why this environment is recommended",
    "alternatives": ["alternative_env_1", "alternative_env_2"],
    "detected_frameworks": ["pytorch", "tensorflow", "jax", etc.],
    "detected_imports": ["torch", "tensorflow", "transformers", etc.]
  }},
  "recommendations": [
    "Suggestion for improving parameter handling"
  ]
}}

Be thorough - check for:
- argparse.ArgumentParser() definitions
- os.environ.get() calls
- Global constants (ALL_CAPS or regular variables)
- Dataclass definitions
- YAML/JSON config loading
- Function default arguments

IMPORTANT: Set "modifiable": true for ALL detected parameters.

For Conda environment recommendation, analyze:
1. Framework imports (torch, tensorflow, jax, etc.)
2. Library-specific imports (transformers, sklearn, etc.)
3. CUDA/GPU usage patterns
4. Specific version requirements
5. Common ML/DL framework patterns

Provide detailed reasoning for the environment recommendation."""


# =============================================================================
# Regex Patterns for Fallback Analysis
# =============================================================================

ARGPARSE_PATTERNS = [
    # Pattern: parser.add_argument('--epochs', type=int, default=10)
    (r'add_argument\s*\(\s*[\'"](--[\w-]+)[\'"]\s*,\s*type\s*=\s*(\w+)\s*,\s*default\s*=\s*([^,)]+)', 'argparse_typed'),
    # Pattern: parser.add_argument('-e', '--epochs', default=10)
    (r'add_argument\s*\(\s*[\'"]-\w[\'"]\s*,\s*[\'"](--[\w-]+)[\'"]\s*,\s*[^)]*default\s*=\s*([^,)\]]+)', 'argparse_short'),
    # Pattern: parser.add_argument('--epochs', default=10) without type
    (r'add_argument\s*\(\s*[\'"](--[\w-]+)[\'"]\s*,\s*[^)]*default\s*=\s*([^,)\]]+)', 'argparse_simple'),
]

ENV_VAR_PATTERNS = [
    # Pattern: EPOCHS = int(os.environ.get('EPOCHS', 10))
    (r'(\w+)\s*=\s*(?:int|float|str)?\s*\(\s*os\.environ\.get\s*\(\s*[\'"](\w+)[\'"]\s*,\s*([^)]+)\)\s*\)', 'env_var_typed'),
    # Pattern: EPOCHS = os.getenv('EPOCHS', '10')
    (r'(\w+)\s*=\s*os\.getenv\s*\(\s*[\'"](\w+)[\'"]\s*,\s*([^)]+)\)', 'env_var_getenv'),
]

HARDCODED_PATTERNS = [
    # Pattern with type hint: EPOCHS: int = 10
    (r'^(\s*)([A-Z_][A-Z0-9_]*)\s*:\s*(?:int|float|str)?\s*=\s*([^#\n]+)', 'typed_assignment'),
    # Pattern: EPOCHS = 10
    (r'^(\s*)([A-Z_][A-Z0-9_]*)\s*=\s*([^#\n]+)', 'simple_assignment'),
]

# Common training parameter names to look for
TRAINING_PARAM_NAMES = {
    # Epochs and steps
    'epochs', 'epoch', 'num_epochs', 'max_epochs', 'steps', 'max_steps', 'num_steps',
    'warmup_steps', 'warmup_epochs', 'gradient_accumulation_steps', 'eval_steps',
    # Batch sizes
    'batch_size', 'batch', 'train_batch_size', 'eval_batch_size', 'per_device_batch_size',
    'micro_batch_size', 'global_batch_size', 'effective_batch_size',
    # Learning rate
    'lr', 'learning_rate', 'base_lr', 'init_lr', 'peak_lr', 'min_lr', 'max_lr',
    'warmup_lr', 'encoder_lr', 'decoder_lr', 'backbone_lr', 'head_lr',
    # Optimizer
    'weight_decay', 'momentum', 'beta1', 'beta2', 'eps', 'optimizer', 'scheduler',
    'lr_scheduler', 'lr_decay', 'gamma', 'step_size', 'patience', 'factor',
    # Model architecture
    'hidden_dim', 'hidden_size', 'embed_dim', 'd_model', 'd_head', 'd_ff',
    'num_layers', 'num_blocks', 'depth', 'num_heads', 'num_attention_heads',
    'head_dim', 'intermediate_size', 'vocab_size', 'num_classes', 'num_labels',
    'num_channels', 'in_channels', 'out_channels', 'kernel_size', 'dropout',
    'dropout_rate', 'attention_dropout', 'activation', 'hidden_act',
    'max_seq_len', 'max_sequence_length', 'sequence_length', 'seq_len',
    'max_position_embeddings', 'max_length', 'block_size', 'context_window',
    # Data
    'train_split', 'val_split', 'test_split', 'validation_split', 'split_ratio',
    'num_workers', 'pin_memory', 'shuffle', 'drop_last', 'augmentation',
    # Hardware
    'device', 'gpu', 'gpus', 'num_gpus', 'cuda', 'fp16', 'bf16', 'mixed_precision',
    'seed', 'random_seed', 'deterministic', 'compile', 'compile_mode',
    # Paths
    'data_path', 'data_dir', 'dataset_path', 'output_dir', 'save_dir', 'log_dir',
    'checkpoint_dir', 'model_dir', 'cache_dir', 'pretrained_model', 'model_name_or_path',
    # Logging
    'log_interval', 'logging_steps', 'eval_interval', 'eval_frequency',
    'save_interval', 'save_steps', 'print_freq', 'report_to', 'run_name',
    # Training settings
    'gradient_clip', 'clip_grad', 'max_grad_norm', 'early_stopping', 'patience',
    'metric_for_best_model', 'greater_is_better', 'load_best_at_end',
}


# =============================================================================
# Core Analyzer Class
# =============================================================================

class ParameterExtractor:
    """Universal parameter extractor for ML training scripts"""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm_client = None
        
        if use_llm:
            self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize LLM client (OpenAI, local, or None)"""
        # Try OpenAI
        try:
            import openai
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                self.llm_client = ('openai', openai)
                return
        except ImportError:
            pass
        
        # Try local LLM (ollama)
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                self.llm_client = ('ollama', 'http://localhost:11434')
                return
        except:
            pass
        
        self.llm_client = None
    
    def analyze_script(self, script_path: str, script_content: str = None) -> Dict:
        """
        Main entry point - analyze a training script and extract all parameters
        
        Args:
            script_path: Path to the Python script
            script_content: Optional pre-loaded content
            
        Returns:
            Dict with parameters, main_entrypoint, and recommendations
        """
        if script_content is None:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        
        # Try LLM analysis first if available
        if self.use_llm and self.llm_client:
            try:
                result = self._llm_analyze(script_content)
                if result and result.get('parameters'):
                    print("[LLM] Successfully analyzed with AI")
                    return self._normalize_result(result, script_path)
            except Exception as e:
                print(f"[WARN] LLM analysis failed: {e}")
        
        # Fallback to regex analysis
        print("[REGEX] Using pattern matching analysis...")
        result = self._regex_analyze(script_content, script_path)
        return self._normalize_result(result, script_path)
    
    def _llm_analyze(self, script_content: str) -> Dict:
        """Use LLM to intelligently analyze the script"""
        client_type, client = self.llm_client
        
        if client_type == 'openai':
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(script_content=script_content[:8000])}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content
        elif client_type == 'ollama':
            import requests
            response = requests.post(
                f"{client}/api/generate",
                json={
                    "model": "codellama",
                    "prompt": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(script_content=script_content[:6000])}",
                    "stream": False
                },
                timeout=60
            )
            content = response.json()['response']
        else:
            raise ValueError("Unknown LLM client type")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in LLM response")
    
    def _regex_analyze(self, script_content: str, script_path: str) -> Dict:
        """Fallback regex-based analysis"""
        parameters = {}
        
        # Extract argparse parameters
        self._extract_argparse_params(script_content, parameters)
        
        # Extract environment variable parameters
        self._extract_env_params(script_content, parameters)
        
        # Extract hardcoded parameters
        self._extract_hardcoded_params(script_content, parameters)
        
        return {
            'parameters': parameters,
            'main_entrypoint': {
                'script': os.path.basename(script_path),
                'function': self._detect_main_function(script_content),
                'has_argparse': 'ArgumentParser' in script_content,
                'has_config_file': any(x in script_content for x in ['yaml', 'json.load', 'OmegaConf'])
            },
            'recommendations': [
                "Consider converting hardcoded parameters to argparse for easier tuning",
                "Add --seed argument for reproducibility"
            ] if any(p.get('source') == 'hardcoded' for p in parameters.values()) else []
        }
    
    def _extract_argparse_params(self, content: str, parameters: Dict):
        """Extract argparse-based parameters"""
        for pattern, pattern_type in ARGPARSE_PATTERNS:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if pattern_type == 'argparse_typed':
                    arg_name, param_type, default = match.groups()
                    param_name = arg_name.lstrip('-').replace('-', '_')
                    parameters[param_name] = {
                        'type': self._infer_type(param_type.strip(), default.strip()),
                        'default': self._parse_default(default.strip()),
                        'description': f'{param_name} parameter',
                        'source': 'argparse',
                        'arg_name': arg_name,
                        'modifiable': True
                    }
    
    def _extract_env_params(self, content: str, parameters: Dict):
        """Extract environment variable parameters"""
        for pattern, pattern_type in ENV_VAR_PATTERNS:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                var_name, env_var, default = match.groups()
                param_name = var_name.lower()
                parameters[param_name] = {
                    'type': self._infer_type(None, default.strip()),
                    'default': self._parse_default(default.strip()),
                    'description': f'{param_name} (environment variable)',
                    'source': 'env_var',
                    'env_var': env_var,
                    'var_name': var_name,
                    'modifiable': True
                }
    
    def _extract_hardcoded_params(self, content: str, parameters: Dict):
        """Extract hardcoded parameters (constants)"""
        # Look for common training parameter patterns
        for param_name in TRAINING_PARAM_NAMES:
            # Pattern: VAR_NAME = value (case-insensitive, but capture actual case)
            pattern = rf'^(\s*)({re.escape(param_name)})\s*=\s*([^#\n]+)'
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                actual_var_name = match.group(2)  # Capture the actual case used in code
                value_str = match.group(3).strip()
                if param_name not in parameters:  # Don't override if already detected
                    parameters[param_name] = {
                        'type': self._infer_type(None, value_str),
                        'default': self._parse_default(value_str),
                        'description': f'{param_name} hyperparameter',
                        'source': 'hardcoded',
                        'var_name': actual_var_name,  # Use actual case from code
                        'modifiable': True
                    }
    
    def _detect_main_function(self, content: str) -> Optional[str]:
        """Detect the main training function"""
        if 'def main(' in content:
            return 'main'
        elif 'def train(' in content:
            return 'train'
        return None
    
    def _infer_type(self, declared_type: str, value_str: str) -> str:
        """Infer parameter type from declaration or value"""
        if declared_type:
            type_map = {'int': 'int', 'float': 'float', 'str': 'str', 'bool': 'bool'}
            return type_map.get(declared_type.lower(), 'str')
        
        value_str = value_str.strip().strip('"\'')
        
        if value_str in ['True', 'False', 'true', 'false']:
            return 'bool'
        try:
            int(value_str)
            return 'int'
        except:
            pass
        try:
            float(value_str)
            return 'float'
        except:
            pass
        return 'str'
    
    def _parse_default(self, value_str: str) -> Any:
        """Parse default value string to Python object"""
        value_str = value_str.strip().strip('"\'')
        
        if value_str in ['True', 'true']:
            return True
        if value_str in ['False', 'false']:
            return False
        if value_str in ['None', 'none']:
            return None
        
        try:
            return int(value_str)
        except:
            pass
        try:
            return float(value_str)
        except:
            pass
        
        return value_str
    
    def _normalize_result(self, result: Dict, script_path: str) -> Dict:
        """Normalize and validate analysis result"""
        # Ensure all parameters have required fields
        for param_name, param_info in result.get('parameters', {}).items():
            param_info.setdefault('type', 'str')
            param_info.setdefault('default', None)
            param_info.setdefault('description', f'{param_name} parameter')
            param_info.setdefault('source', 'hardcoded')
            param_info.setdefault('modifiable', True)
        
        # Ensure main_entrypoint exists
        if 'main_entrypoint' not in result:
            result['main_entrypoint'] = {
                'script': os.path.basename(script_path),
                'function': None,
                'has_argparse': False,
                'has_config_file': False
            }
        
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def smart_analyze_script(script_path: str, use_llm: bool = True) -> Dict:
    """
    Convenience function to analyze a training script
    
    Args:
        script_path: Path to Python training script
        use_llm: Whether to try LLM analysis (falls back to regex if unavailable)
    
    Returns:
        Dictionary with extracted parameters and metadata
    """
    extractor = ParameterExtractor(use_llm=use_llm)
    return extractor.analyze_script(script_path)


def analyze_and_print(script_path: str):
    """Analyze script and print formatted results"""
    result = smart_analyze_script(script_path)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS RESULT: {script_path}")
    print(f"{'='*60}")
    
    params = result.get('parameters', {})
    print(f"\nTotal Parameters: {len(params)}\n")
    
    # Group by source
    by_source = {}
    for name, info in params.items():
        source = info.get('source', 'unknown')
        by_source.setdefault(source, []).append((name, info))
    
    for source, items in by_source.items():
        print(f"\n[{source.upper()}] ({len(items)} params)")
        for name, info in items:
            print(f"  {name}: {info.get('default')} ({info.get('type')})")
    
    print(f"\n{'='*60}")
    return result


# =============================================================================
# Main Entry Point (for testing)
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training script parameters')
    parser.add_argument('script', help='Path to training script')
    parser.add_argument('--no-llm', action='store_true', help='Use regex only')
    parser.add_argument('--output', '-o', help='Output JSON file')
    
    args = parser.parse_args()
    
    result = analyze_and_print(args.script)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n[OK] Results saved to: {args.output}")
