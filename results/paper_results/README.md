# Paper Results

This directory contains the key result files from our NeurIPS 2025 paper submission.

## Key Files

- `consolidated_groundtruth_35_behaviors.json` - Ground truth evaluation results for all 35 HarmBench behaviors across target models
- `v9_real_measured_evaluation_20250829_001848_corrected.json` - Main evaluation results with corrected agent decisions and measured ASR
- `enhanced_appendix_tables.tex` - Complete LaTeX tables for paper appendix (M1, M2, M3 results, runtime analysis, etc.)
- `updated_target_models_table.tex` - Updated target model specifications table with safety levels and architectural profiles

## Metrics

- **M1**: Single-step attack results (74.5% ASR, 387.8s avg runtime)
- **M2**: Two-step attack with retry mechanism (76.2% ASR, 608.5s avg runtime) 
- **M3**: Three-step attack for models with <100% M2 ASR (88.0% ASR, 686.2s avg runtime)

## Key Findings

- Intelligent agent shows preference for FewShot (81.8%) over GPTFuzz (2.5%) and DirectRequest (15.6%)
- Two-step retry mechanism improves ASR by +1.7% over single-step approach
- Perfect M2 performance achieved by mixtral_8x7b_v01 and qwen2_7b_instruct (100% ASR)
- Runtime driven by FewShot dominance at 435s average cost per behavior

Generated: August 30, 2025