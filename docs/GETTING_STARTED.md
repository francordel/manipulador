# Getting Started with Manipulador

This guide will help you get up and running with Manipulador, an agentic red-teaming framework for LLM vulnerability discovery.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for best performance)
- 32GB+ RAM for large model evaluations
- ~100GB disk space for models and datasets

## Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/Manipulador-OSS.git
cd Manipulador-OSS

# Create virtual environment
python -m venv manipulador-env
source manipulador-env/bin/activate  # On Windows: manipulador-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Install with pip

```bash
pip install manipulador
```

## Quick Start

### 1. Basic Agent Evaluation

```python
from agents.behavior_attack_selector_real_2step import RealTwoStepBehaviorAttackSelector

# Initialize the agentic selector
agent = RealTwoStepBehaviorAttackSelector()

# Select attack method for a behavior  
method = agent.select_attack_method(
    model_name="mistral_7b_v03_medium",
    behavior_description="Generate misinformation about climate change",
    category="misinformation"
)

print(f"Selected method: {method}")
# Output: FewShot, DirectRequest, or GPTFuzz

# Perform 2-step evaluation with retry mechanism
result = agent.perform_two_step_evaluation(
    model_name="mistral_7b_v03_medium",
    behavior_description="Generate misinformation about climate change", 
    category="misinformation"
)

print(f"Step 1: {result['step1_method']}")
print(f"Final: {result['final_method']}")  
print(f"Used Retry: {result['used_retry']}")
```

### 2. Run Full Evaluation

```bash
# Evaluate agentic selector on all behaviors and models
python scripts/run_comprehensive_evaluation.py \
    --agent_type agentic_2step \
    --models_config configs/model_configs/models.yaml \
    --behaviors_file data/behavior_datasets/harmbench_behaviors_text_test.csv \
    --output_dir results/my_evaluation
```

### 3. Generate Ground Truth

```bash
# Generate baseline method results for comparison
python scripts/generate_ground_truth.py \
    --methods directrequest,fewshot,gptfuzz \
    --models_config configs/model_configs/models.yaml \
    --output_dir data/ground_truth
```

## Configuration

### Model Configuration

Edit `configs/model_configs/models.yaml` to configure target models:

```yaml
mistral_7b_v03_medium:
  model_name_or_path: "mistralai/Mistral-7B-Instruct-v0.3"
  dtype: "float16"
  num_gpus: 1
  safety_level: "medium"
  context_window: 32768
  
qwen2_7b_instruct:
  model_name_or_path: "Qwen/Qwen2-7B-Instruct"
  dtype: "float16" 
  num_gpus: 1
  safety_level: "medium"
  context_window: 131072
```

### Attack Method Configuration

Configure attack methods in `configs/method_configs/`:

```yaml
# FewShot_config.yaml
method_name: "FewShot"
num_fewshot_examples: 3
mutator_model: "openchat/openchat-3.5-1210"
max_attempts: 5
temperature: 0.7
```

## Understanding Results

### Agent Results Format

```json
{
  "performance_metrics": {
    "success_rate": 64.3,
    "accuracy": 53.1, 
    "total_behaviors": 35,
    "successful_choices": 27,
    "optimal_choices": 17
  },
  "method_effectiveness": {
    "FewShot": {"chosen": 25, "successful": 24, "success_rate": 96.0},
    "GPTFuzz": {"chosen": 8, "successful": 6, "success_rate": 75.0},
    "DirectRequest": {"chosen": 2, "successful": 1, "success_rate": 50.0}
  }
}
```

### Key Metrics

- **Success Rate**: Percentage of behaviors where the selected method succeeded
- **Accuracy**: Percentage of behaviors where optimal method was selected  
- **Method Distribution**: How often each attack method was chosen
- **Optimal vs Suboptimal**: Quality of method selection decisions

## Next Steps

1. **Reproduce Paper Results**: Follow the [Reproducibility Guide](REPRODUCIBILITY.md)
2. **Custom Evaluation**: See [examples/custom_evaluation.py](../examples/custom_evaluation.py)
3. **API Reference**: Check [API_REFERENCE.md](API_REFERENCE.md) for detailed documentation
4. **Evaluation Guide**: Read [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for evaluation methodology

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use smaller models
2. **Model loading fails**: Check model paths in configuration files
3. **Slow evaluation**: Use GPU acceleration and reduce number of behaviors

### Getting Help

- **GitHub Issues**: [Report bugs and ask questions](https://github.com/your-org/Manipulador-OSS/issues)
- **Discussions**: [Community discussions](https://github.com/your-org/Manipulador-OSS/discussions)
- **Email**: anonymous@example.com

## Performance Expectations

| Configuration | Expected Runtime | Memory Usage | ASR |
|---------------|------------------|--------------|-----|
| Agentic Single-step (35 behaviors) | ~2 hours | 16GB GPU | 35.2% |
| Agentic Two-step (35 behaviors) | ~3 hours | 16GB GPU | 64.3% |  
| Full evaluation (350 behaviors) | ~20 hours | 32GB GPU | Variable |

*Times are for single A100 80GB GPU. Scale accordingly for your hardware.*