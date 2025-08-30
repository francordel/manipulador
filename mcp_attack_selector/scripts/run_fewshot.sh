#!/bin/bash

# FewShot Attack Execution Script
# This script runs the FewShot attack method against a target model

set -e  # Exit on any error

# Script parameters
MODEL_NAME="$1"
BEHAVIOR_FILE="$2"
RESULTS_DIR="$3"
N_SHOTS="${4:-5}"
SAMPLE_SIZE="${5:-1000}"
TIMEOUT_MINUTES="${6:-60}"

# Validate required parameters
if [ -z "$MODEL_NAME" ] || [ -z "$BEHAVIOR_FILE" ] || [ -z "$RESULTS_DIR" ]; then
    echo "Usage: $0 <model_name> <behavior_file> <results_dir> [n_shots] [sample_size] [timeout_minutes]"
    echo "Example: $0 openchat_3_5_1210 ./behaviors/five_behaviors.txt ./results 5 1000 60"
    exit 1
fi

# Display execution parameters
echo "🚀 [FewShot] Starting FewShot attack execution"
echo "📊 Model: $MODEL_NAME"
echo "📂 Behavior file: $BEHAVIOR_FILE"
echo "💾 Results directory: $RESULTS_DIR"
echo "🎯 N-shots: $N_SHOTS"
echo "📏 Sample size: $SAMPLE_SIZE"
echo "⏱️  Timeout: $TIMEOUT_MINUTES minutes"
echo "----------------------------------------"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Get the project root directory (assuming this script is in mcp_attack_selector/scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "📁 Project root: $PROJECT_ROOT"

# Change to project root for execution
cd "$PROJECT_ROOT"

# Set up environment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if behavior file exists
if [ ! -f "$BEHAVIOR_FILE" ]; then
    echo "❌ Error: Behavior file not found: $BEHAVIOR_FILE"
    exit 1
fi

# Execute the FewShot attack
echo "🎯 Executing FewShot attack..."

# Run the attack with proper timeout
timeout "${TIMEOUT_MINUTES}m" python scripts/run_pipeline.py \
    --step all \
    --mode local \
    --methods FewShot \
    --models "$MODEL_NAME" \
    --behaviors_path "$BEHAVIOR_FILE" \
    --save_dir "$RESULTS_DIR" \
    --fewshot_n_shots "$N_SHOTS" \
    --fewshot_sample_size "$SAMPLE_SIZE" \
    --fewshot_num_steps 5 \
    2>&1 | tee "$RESULTS_DIR/fewshot_execution.log"

# Check execution status
EXECUTION_STATUS=$?

if [ $EXECUTION_STATUS -eq 0 ]; then
    echo "✅ FewShot attack completed successfully"
    echo "📊 Results saved to: $RESULTS_DIR"
    
    # Display basic result summary if available
    if [ -f "$RESULTS_DIR/results_summary.json" ]; then
        echo "📈 Result summary:"
        cat "$RESULTS_DIR/results_summary.json" | head -10
    fi
    
    # Look for FewShot-specific results
    if [ -f "$RESULTS_DIR/fewshot_results.json" ]; then
        echo "📊 FewShot specific results:"
        cat "$RESULTS_DIR/fewshot_results.json" | head -10
    fi
    
elif [ $EXECUTION_STATUS -eq 124 ]; then
    echo "⏰ FewShot attack timed out after $TIMEOUT_MINUTES minutes"
    exit 124
else
    echo "❌ FewShot attack failed with exit code: $EXECUTION_STATUS"
    echo "📋 Check logs in: $RESULTS_DIR/fewshot_execution.log"
    exit $EXECUTION_STATUS
fi

# Display final status
echo "----------------------------------------"
echo "🏁 FewShot attack execution completed"
echo "📊 Model: $MODEL_NAME"
echo "🎯 Shots used: $N_SHOTS"
echo "📏 Sample size: $SAMPLE_SIZE"
echo "📂 Results: $RESULTS_DIR"
echo "🕐 Execution time: $((SECONDS / 60)) minutes and $((SECONDS % 60)) seconds"