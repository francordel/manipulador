# MANIPULADOR - Advanced Language Model Security Assessment Framework

![Manipulador](Manipulador.png)

[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)](https://github.com)

**Manipulador** is a modular agentic red teaming framework designed for systematic LLM vulnerability discovery. It features intelligent attack selection, multi-model evaluation, and comprehensive analysis tools for security research.

## Key Features

- **Intelligent Agent System**: Agentic selector with 2-step retry mechanism and model-adaptive selection
- **Multi-Attack Framework**: DirectRequest, FewShot, and GPTFuzz attack methods
- **Comprehensive Evaluation**: Clean separation between decision-making and scoring for research validity
- **Ground Truth Integration**: Measured ASR using real experimental results
- **Performance Analytics**: Detailed metrics, retry analysis, and LaTeX table generation
- **Research Valid**: No ground truth cheating, academically publishable results

## Performance Highlights

- **Manipulador1 (1-step):** 74.5% ASR, 387.7s avg runtime  
- **Manipulador2 (2-step):** 76.3% ASR, 608.5s avg runtime  
- **Oracle (M3):** 88.0% ASR, 686.2s avg runtime  
- **DirectRequest:** 35.9% ASR, 13.1s avg runtime  
- **FewShot:** 76.3% ASR, 435.3s avg runtime  
- **GPTFuzz:** 32.7% ASR, 1170.3s avg runtime  
- Evaluated across multiple instruction-tuned LLMs from Mistral, Qwen, Mixtral, Granite, and Llama families  

## Project Structure

```
Manipulador-OSS/
├── README.md                        # This file
├── REPRODUCTION_GUIDE.md            # Quick reproduction guide
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package installation
├── LICENSE                         # MIT License
├── baselines/                      # Core attack implementations
│   ├── direct_request/            # DirectRequest baseline
│   ├── fewshot/                   # FewShot attack method
│   ├── gptfuzz/                   # GPTFuzz implementation
│   └── model_utils.py             # Model utilities
├── agents/                         # Intelligent agent implementations
│   ├── behavior_attack_selector_generic.py      # Model-adaptive agent
│   ├── behavior_attack_selector_real_2step.py   # Retry-enabled agent
│   └── ground_truth_loader.py     # Ground truth utilities
├── mcp_attack_selector/            # MCP server implementation
├── configs/                        # Configuration files
│   ├── model_configs/             # Target model configurations
│   ├── method_configs/            # Attack method configurations
│   └── tools/                     # Agent tool configurations
├── data/                          # Datasets and evaluation data
│   ├── behavior_datasets/         # HarmBench behavior datasets
│   └── updated_35_behaviors_complete.json
├── scripts/                       # Evaluation and utility scripts
│   ├── run_real_measured_evaluation.py    # M1 evaluation
│   ├── run_real_2step_evaluation.py       # M2 evaluation  
│   └── analyze_results.py                # Results analysis
├── results/                       # Results and analysis
│   ├── paper_results/            # Key paper results (JSON, LaTeX tables)
│   └── agent_results/            # Sample evaluation outputs
├── docs/                         # Documentation
│   ├── GETTING_STARTED.md        # Quick start guide
│   └── REPRODUCIBILITY.md        # Detailed reproduction steps
└── examples/                     # Usage examples
    ├── basic_usage.py            # Simple usage example
    └── custom_evaluation.py     # Custom evaluation setup
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Manipulador-OSS.git
cd Manipulador-OSS

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

### Basic Usage
```

```python
from agents.behavior_attack_selector_generic import GenericV9BehaviorAttackSelector
from baselines.model_utils import load_model

# Initialize the intelligent agent
agent = GenericV9BehaviorAttackSelector()

# Select attack method for a behavior
method = agent.select_attack_method(
    model_name="mistral_7b_v03_medium",
    behavior_description="Generate harmful content about...",
    category="harmful_content"
)

print(f"Selected method: {method}")
```

### Running Full Evaluation

```bash
# Run comprehensive evaluation across all models and behaviors
python scripts/run_comprehensive_evaluation.py \
    --agent_type agentic_2step \
    --models_config configs/model_configs/models.yaml \
    --behaviors_file data/behavior_datasets/harmbench_behaviors_text_test.csv \
    --output_dir results/evaluation_$(date +%Y%m%d_%H%M%S)
```

## Reproducing Paper Results

**See [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) for detailed instructions**

### Quick View of Results
All key results are in `results/paper_results/`:
- **Ground truth**: `consolidated_groundtruth_35_behaviors.json`
- **Main evaluation**: `v9_real_measured_evaluation_20250829_001848_corrected.json` 
- **LaTeX tables**: `enhanced_appendix_tables.tex`

### Run New Evaluations
```bash
# M1 (Single-step) 
python scripts/run_real_measured_evaluation.py

# M2 (Two-step with retry)
python scripts/run_real_2step_evaluation.py

# Note: Evaluating all three baseline methods across the full dataset would require 
# 142 GPU-hours (DirectRequest: 1.1 hours, FewShot: 38.1 hours, GPTFuzz: 102.4 hours), 
# making exhaustive evaluation prohibitively expensive. All experiments use four NVIDIA A100 80GB GPUs.
```

## Evaluation Results

### Attack Success Rates

| Method | ASR (%) | Runtime (s) | Description |
|--------|---------|-------------|-------------|
| DirectRequest | 35.9 | 13.1 | Simple baseline |
| FewShot | 76.3 | 435.3 | Few-shot prompting |
| GPTFuzz | 32.7 | 1170.3 | Iterative mutations |
| **Manipulador1 (1-step)** | **74.5** | **387.7** | **Intelligent selection** |
| **Manipulador2 (2-step)** | **76.3** | **608.5** | **With retry mechanism** |
| **Oracle (M3)** | **88.0** | **686.2** | **Sequential execution of all attacks** |

### Method Selection Distribution

The agentic selector demonstrates intelligent selection behavior:
- **FewShot**: 73% (primary method)
- **GPTFuzz**: 22% (secondary choice) 
- **DirectRequest**: 5% (minimal usage)

## Configuration

### Target Models

Configure target models in `configs/model_configs/models.yaml`:

```yaml
mistral_7b_v03_medium:
  model_name_or_path: "mistralai/Mistral-7B-Instruct-v0.3"
  dtype: "float16"
  num_gpus: 1
  safety_level: "medium"
  context_window: 32768
```

### Attack Methods

Configure attack parameters in `configs/method_configs/`:

```yaml
# FewShot_config.yaml
method_name: "FewShot"
num_fewshot_examples: 3
mutator_model: "openchat/openchat-3.5-1210"
max_attempts: 5
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built upon the [HarmBench](https://github.com/centerforaisafety/HarmBench) evaluation framework
- Integrates attack methods from GPTFuzz, FewShot, and DirectRequest baselines
- Evaluation infrastructure adapted from the HarmBench codebase
