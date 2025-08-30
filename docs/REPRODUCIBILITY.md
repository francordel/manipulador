# Reproducibility Guide

This document provides detailed instructions for reproducing the results presented in our NeurIPS 2025 paper: "Manipulador: An Agentic Red-Teaming Framework for LLM Vulnerability Discovery".

## Hardware Requirements

### Recommended Configuration
- **GPUs**: 4×A100 80GB or equivalent
- **RAM**: 128GB system memory
- **Storage**: 500GB free space
- **CPU**: 32+ cores recommended

### Minimum Configuration  
- **GPUs**: 2×RTX 4090 24GB or 1×A100 40GB
- **RAM**: 64GB system memory
- **Storage**: 200GB free space
- **CPU**: 16+ cores

## Environment Setup

### 1. System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev
sudo apt install -y build-essential cmake git curl

# Install CUDA 11.8 (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### 2. Python Environment

```bash
# Create isolated environment
python3.9 -m venv neurips2025-env
source neurips2025-env/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install Manipulador
pip install -r requirements.txt
pip install -e .
```

### 3. Model Downloads

```bash
# Download required models (will take ~2 hours)
python scripts/download_models.py --models configs/model_configs/neurips2025_models.yaml
```

## Dataset Preparation

### 1. HarmBench Behaviors

```bash
# Download and prepare HarmBench dataset
python scripts/prepare_harmbench_data.py \
    --output_dir data/behavior_datasets/ \
    --subset test \
    --num_behaviors 35
```

### 2. Ground Truth Generation

This step generates baseline results for DirectRequest, FewShot, and GPTFuzz methods:

```bash
# Generate ground truth for all models (8-12 hours on 4×A100)
python scripts/generate_ground_truth.py \
    --methods directrequest,fewshot,gptfuzz \
    --models_config configs/model_configs/neurips2025_models.yaml \
    --behaviors_file data/behavior_datasets/harmbench_behaviors_text_test.csv \
    --output_dir data/ground_truth/ \
    --batch_size 4 \
    --num_workers 4
```

Expected ground truth ASRs:
| Method | Avg ASR | Runtime (per behavior) |
|--------|---------|----------------------|
| DirectRequest | 7.7% | 13.1s |
| FewShot | 50.6% | 435.3s |
| GPTFuzz | 44.3% | 1170.3s |

## Main Experiments

### 1. V9 Single-Step Agent Evaluation

```bash
# Evaluate V9 agent without retry mechanism
python scripts/run_comprehensive_evaluation.py \
    --agent_type v9_generic \
    --models_config configs/model_configs/neurips2025_models.yaml \
    --behaviors_file data/behavior_datasets/harmbench_behaviors_text_test.csv \
    --output_dir results/v9_single_step/ \
    --batch_size 1 \
    --save_intermediate
```

**Expected Results:**
- Success Rate: 35.2%
- Method Distribution: FewShot 73%, GPTFuzz 22%, DirectRequest 5%
- Runtime: ~180 seconds per behavior

### 2. V9 Two-Step Agent Evaluation

```bash  
# Evaluate V9 agent with retry mechanism
python scripts/run_comprehensive_evaluation.py \
    --agent_type v9_2step \
    --models_config configs/model_configs/neurips2025_models.yaml \
    --behaviors_file data/behavior_datasets/harmbench_behaviors_text_test.csv \
    --output_dir results/v9_two_step/ \
    --batch_size 1 \
    --enable_retry \
    --save_intermediate
```

**Expected Results:**
- Success Rate: 64.3% (+29.1% improvement)
- Runtime: ~180 seconds per behavior (similar to single-step)
- Retry Success Rate: 45.2% of failed first attempts

### 3. Cross-Model Analysis

```bash
# Run evaluation across all 10 target models
for model in mistral_7b_v03 mixtral_8x7b qwen2_7b qwen2_5_0_5b qwen2_1_5b granite_3_3_8b granite_3_3_2b llama_3_2_1b llama_3_2_3b meta_llama_3_70b; do
    python scripts/run_comprehensive_evaluation.py \
        --agent_type v9_2step \
        --target_models $model \
        --behaviors_file data/behavior_datasets/harmbench_behaviors_text_test.csv \
        --output_dir results/cross_model_analysis/$model/ \
        --batch_size 1
done
```

## Analysis and Visualization

### 1. Generate Performance Tables

```bash
# Create LaTeX tables for paper
python scripts/analyze_results.py \
    --results_dirs results/v9_single_step/ results/v9_two_step/ \
    --ground_truth_dir data/ground_truth/ \
    --output_dir results/analysis/ \
    --generate_latex_tables
```

### 2. Cross-Model Analysis

```bash
# Analyze performance across models
python scripts/cross_model_analysis.py \
    --results_dir results/cross_model_analysis/ \
    --output_dir results/analysis/ \
    --generate_plots
```

### 3. Method Selection Analysis

```bash
# Analyze method selection patterns
python scripts/method_selection_analysis.py \
    --agent_results results/v9_two_step/ \
    --ground_truth_dir data/ground_truth/ \
    --output_dir results/analysis/ \
    --create_visualizations
```

## Expected Runtime

| Experiment | Hardware | Time |
|------------|----------|------|
| Ground Truth Generation | 4×A100 80GB | 8-12 hours |
| V9 Single-Step (10 models) | 4×A100 80GB | 4-6 hours |
| V9 Two-Step (10 models) | 4×A100 80GB | 4-6 hours |
| **Total** | **4×A100 80GB** | **16-24 hours** |

With minimum hardware (2×RTX 4090), expect 2-3x longer runtime.

## Verification

### 1. Check Key Metrics

After running all experiments, verify these key results match paper Table 2:

```bash
python scripts/verify_results.py \
    --results_dir results/ \
    --ground_truth_dir data/ground_truth/ \
    --expected_metrics configs/expected_results.json
```

**Expected Output:**
```
✓ V9 Single-Step ASR: 35.2% (±1.5%)
✓ V9 Two-Step ASR: 64.3% (±1.8%) 
✓ Improvement: +29.1% (±2.0%)
✓ FewShot Selection: 73% (±3%)
✓ GPTFuzz Selection: 22% (±3%)
✓ DirectRequest Selection: 5% (±2%)
```

### 2. Statistical Significance

```bash
# Run statistical tests
python scripts/statistical_analysis.py \
    --results_dir results/ \
    --output_dir results/analysis/statistics/ \
    --confidence_level 0.95
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Reduce batch size
   --batch_size 1
   
   # Use gradient checkpointing
   --gradient_checkpointing
   
   # Use smaller models for testing
   --target_models mistral_7b_v03,qwen2_7b
   ```

2. **Model Loading Failures**
   ```bash
   # Check model paths
   python scripts/verify_models.py --models_config configs/model_configs/neurips2025_models.yaml
   
   # Re-download specific model
   python scripts/download_models.py --models mistral_7b_v03 --force
   ```

3. **Slow Evaluation**
   ```bash
   # Enable mixed precision
   --fp16
   
   # Increase parallel workers
   --num_workers 8
   
   # Use smaller behavior subset for testing
   --max_behaviors 10
   ```

### Performance Debugging

```bash
# Profile GPU usage
python scripts/profile_evaluation.py \
    --agent_type v9_2step \
    --target_models mistral_7b_v03 \
    --max_behaviors 5 \
    --profile_gpu
```

## Reproducing Specific Results

### Table 2: Main Results

```bash
# Generate exact Table 2 from paper
python scripts/generate_table2.py \
    --v9_single_results results/v9_single_step/ \
    --v9_two_step_results results/v9_two_step/ \
    --ground_truth_dir data/ground_truth/ \
    --output results/analysis/table2.tex
```

### Figure 3: Method Selection Distribution  

```bash
# Generate method selection visualization
python scripts/generate_figure3.py \
    --results_dir results/v9_two_step/ \
    --output results/analysis/figure3.pdf
```

### Table 3: Cross-Model Performance

```bash
# Generate cross-model comparison table
python scripts/generate_table3.py \
    --results_dir results/cross_model_analysis/ \
    --output results/analysis/table3.tex
```

## Contact

If you encounter issues reproducing results:

1. **Check GitHub Issues**: [Known issues and solutions](https://github.com/your-org/Manipulador-OSS/issues)
2. **Create Issue**: [Report reproduction problems](https://github.com/your-org/Manipulador-OSS/issues/new)  
3. **Email**: anonymous@example.com with "[Manipulador Reproduction]" in subject

## Citation

If you use this code to reproduce our results, please cite:

```bibtex
@inproceedings{anonymized2025manipulador,
  title={Manipulador: An Agentic Red-Teaming Framework for LLM Vulnerability Discovery},
  author={Anonymous Authors},
  booktitle={Advances in Neural Information Processing Systems},  
  year={2025}
}
```