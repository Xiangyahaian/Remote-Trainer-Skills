# Conda Environment Intelligence Feature

## 功能概述

**象牙海岸** 为 remote-training skill 添加了 **Conda 环境智能推荐** 功能。这是一个重要功能，让 LLM 在初始化项目时自动分析代码并推荐最适合的 Conda 环境。

## 核心功能

### 1. 自动环境检测
```python
# LLM 分析代码中的以下内容：
- 导入语句 (import torch, import tensorflow)
- 框架使用模式 (torch.nn, torch.optim)
- CUDA/GPU 检测代码
- 特定库的使用 (transformers, sklearn, etc.)
```

### 2. 智能推荐
```python
# LLM 返回详细的推荐信息
{
  "conda_environment": {
    "recommended": "pytorch",
    "confidence": "high",
    "reasoning": "详细解释为什么推荐这个环境",
    "alternatives": ["torch", "deep-learning"],
    "detected_frameworks": ["pytorch", "torchvision"],
    "detected_imports": ["torch", "torch.nn", ...]
  }
}
```

### 3. 用户选择流程
**优先级：**
1. **用户指定** (`--conda-env pytorch`) → 直接使用
2. **LLM 推荐** → 用户确认
3. **交互式选择** → 从可用环境列表选择
4. **默认回退** → 使用 `base`

## 使用方式

### 方式 1: 让 LLM 自动推荐
```bash
python scripts/init_project.py /path/to/project
```

**交互示例：**
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

✓ Recommended environment 'pytorch' is AVAILABLE

Use recommended environment 'pytorch'? [Y/n]: 
```

### 方式 2: 直接指定环境（绕过推荐）
```bash
python scripts/init_project.py /path/to/project --conda-env pytorch
```

### 方式 3: 查看推荐但不创建文件
```bash
python scripts/init_project.py /path/to/project --analyze-only
```

## 配置存储

生成的 `config.json` 包含完整的环境信息：
```json
{
  "conda_env": "pytorch",
  "conda_recommendation": {
    "recommended": "pytorch",
    "confidence": "high",
    "reasoning": "Script uses torch.nn, torch.optim, and CUDA detection",
    "alternatives": ["torch", "deep-learning"],
    "detected_frameworks": ["pytorch", "torchvision"],
    "detected_imports": ["torch", "torch.nn", "torch.optim"]
  }
}
```

## 支持的框架检测

| 框架类型 | 检测信号 | 推荐环境 |
|---------|---------|---------|
| PyTorch | `torch`, `torch.nn`, `torch.optim` | `pytorch`, `torch` |
| TensorFlow | `tensorflow`, `keras`, `tf` | `tensorflow`, `tf` |
| Transformers | `transformers`, `datasets` | `transformers`, `huggingface` |
| JAX/Flax | `jax`, `flax`, `optax` | `jax`, `flax` |
| Classic ML | `sklearn`, `numpy`, `pandas` | `base`, `ml`, `scipy` |

## 文件修改清单

1. **`scripts/llm_analyzer.py`**
   - 更新 `USER_PROMPT_TEMPLATE` 添加 Conda 环境检测指令

2. **`scripts/init_project.py`**
   - 新增 `find_available_conda_envs()` 函数
   - 新增 `select_conda_environment()` 函数
   - 更新 `init_project()` 支持 `--conda-env` 参数
   - 更新 `generate_config()` 保存环境信息
   - 更新 `generate_parameters_md()` 显示环境信息

3. **`SKILL.md`**
   - 添加 "Conda Environment Intelligence" 章节
   - 更新所有命令示例
   - 添加配置字段说明
   - 添加故障排除指南

## 重要说明

### 为什么这个功能很重要？

1. **减少配置错误** - 自动检测避免手动选择错误环境
2. **提高效率** - 不再需要手动检查代码来确定使用哪个环境
3. **新手友好** - 帮助不熟悉项目依赖的用户快速上手
4. **文档化** - 将环境选择原因记录在配置中，便于团队协作

### 注意事项

1. **推荐仅供参考** - 用户始终有最终决定权
2. **需要 Conda 安装** - 系统需要安装 Conda 才能检测可用环境
3. **LLM 依赖** - 需要 OpenAI API 或本地 Ollama 来获取智能推荐
4. **可以绕过** - 使用 `--conda-env` 参数可以跳过推荐直接使用指定环境

## 测试命令

```bash
# 测试环境检测（分析模式，不创建文件）
python scripts/init_project.py . --analyze-only

# 使用指定环境初始化
python scripts/init_project.py . --force --conda-env pytorch

# 查看当前配置
cat remote_training/config.json | findstr conda

# 查看参数文档
cat remote_training/PARAMETERS.md
```

---

**功能状态**: ✅ 已实现并集成
**测试状态**: 等待用户测试反馈
**文档状态**: ✅ SKILL.md 已更新
