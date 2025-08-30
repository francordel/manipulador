# Reproduction Guide

This guide helps you reproduce the results from our NeurIPS 2025 paper.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **View paper results**:
   All key results are available in `results/paper_results/`:
   - Ground truth data: `consolidated_groundtruth_35_behaviors.json`
   - Main evaluation: `v9_real_measured_evaluation_20250829_001848_corrected.json`
   - LaTeX tables: `enhanced_appendix_tables.tex`

## Running New Evaluations

### Single-Step Evaluation (M1)
```bash
python scripts/run_real_measured_evaluation.py \
    --agent_type generic \
    --output_file results/new_m1_evaluation.json
```

### Two-Step Evaluation (M2)
```bash
python scripts/run_real_2step_evaluation.py \
    --agent_type real_2step \
    --output_file results/new_m2_evaluation.json
```

### Generate Analysis Tables
```bash
python scripts/analyze_results.py \
    --evaluation_file results/new_evaluation.json \
    --ground_truth_file results/paper_results/consolidated_groundtruth_35_behaviors.json
```

## Core Components

- **Agents**: `agents/behavior_attack_selector_*.py` - Intelligent attack selection
- **Baselines**: `baselines/` - DirectRequest, FewShot, GPTFuzz implementations  
- **Scripts**: `scripts/` - Evaluation and analysis scripts
- **Configs**: `configs/` - Model and method configurations
- **Data**: `data/` - HarmBench behaviors and datasets

## Expected Runtime

This subset is computationally necessary as evaluating all three methods across the full dataset would require 142 GPU-hours (DirectRequest: 1.1 hours, FewShot: 38.1 hours, GPTFuzz: 102.4 hours), making exhaustive evaluation prohibitively expensive. All experiments use the HarmBench database (covering misinformation, harmful instructions, and policy violations) on four NVIDIA A100 80GB GPUs.

- M1 evaluation: Single-step agent evaluation runtime
- M2 evaluation: Two-step agent with retry mechanism runtime  
- M3 analysis: Generated from M2 results, no additional runtime

## Target Models

The evaluation covers multiple LLMs from 0.5B to 46.7B parameters:
- qwen2_5_0_5b_instruct (0.5B, Light safety)
- llama_3_2_1b (1B, Medium safety)  
- qwen2_1_5b_instruct (1.5B, Light safety)
- granite_3_3_2b_instruct (2B, Medium safety)
- llama_3_2_3b_instruct (3B, Medium safety)
- qwen2_7b_instruct (7B, Medium safety)
- mistral_7b_v03_medium (7.3B, Medium safety)
- granite_3_3_8b_instruct (8B, High safety)
- mixtral_8x7b_v01 (46.7B MoE, High safety)

See `results/paper_results/updated_target_models_table.tex` for detailed specifications.