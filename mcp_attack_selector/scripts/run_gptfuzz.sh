#!/bin/bash

# GPTFuzz Attack Execution Script
# This script runs the GPTFuzz attack method against a target model

set -e  # Exit on any error

# Script parameters
MODEL_NAME="$1"
BEHAVIOR_FILE="$2"
RESULTS_DIR="$3"
MUTATION_MODEL="${4:-openchat-3.5}"
DTYPE="${5:-float16}"
MAX_QUERY="${6:-1000}"
MAX_JAILBREAK="${7:-1}"
MAX_ITERATION="${8:-100}"
ENERGY="${9:-1}"
TIMEOUT_MINUTES="${10:-180}"

# Validate required parameters
if [ -z "$MODEL_NAME" ] || [ -z "$BEHAVIOR_FILE" ] || [ -z "$RESULTS_DIR" ]; then
    echo "Usage: $0 <model_name> <behavior_file> <results_dir> [mutation_model] [dtype] [max_query] [max_jailbreak] [max_iteration] [energy] [timeout_minutes]"
    echo "Example: $0 openchat_3_5_1210 ./behaviors/five_behaviors.txt ./results openchat-3.5 float16 1000 1 100 1 180"
    exit 1
fi

# Display execution parameters
echo "🚀 [GPTFuzz] Starting GPTFuzz attack execution"
echo "📊 Model: $MODEL_NAME"
echo "📂 Behavior file: $BEHAVIOR_FILE"
echo "💾 Results directory: $RESULTS_DIR"
echo "🤖 Mutation model: $MUTATION_MODEL"
echo "🔢 Data type: $DTYPE"
echo "🎯 Max queries: $MAX_QUERY"
echo "🔓 Max jailbreaks: $MAX_JAILBREAK"
echo "🔄 Max iterations: $MAX_ITERATION"
echo "⚡ Energy: $ENERGY"
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

# Check GPU memory before execution
echo "🔍 Checking GPU memory status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits
    echo "💾 GPU memory status logged"
else
    echo "⚠️  nvidia-smi not available, skipping GPU check"
fi

# Execute the GPTFuzz attack
echo "🎯 Executing GPTFuzz attack..."

# Run the attack with proper timeout
timeout "${TIMEOUT_MINUTES}m" python scripts/run_pipeline.py \
    --step all \
    --mode local \
    --methods GPTFuzz \
    --models "$MODEL_NAME" \
    --behaviors_path "$BEHAVIOR_FILE" \
    --save_dir "$RESULTS_DIR" \
    --gptfuzz_mutation_model "$MUTATION_MODEL" \
    --gptfuzz_use_local_model \
    --gptfuzz_dtype "$DTYPE" \
    --gptfuzz_max_query "$MAX_QUERY" \
    --gptfuzz_max_jailbreak "$MAX_JAILBREAK" \
    --gptfuzz_max_iteration "$MAX_ITERATION" \
    --gptfuzz_energy "$ENERGY" \
    2>&1 | tee "$RESULTS_DIR/gptfuzz_execution.log"

# Check execution status
EXECUTION_STATUS=$?

if [ $EXECUTION_STATUS -eq 0 ]; then
    echo "✅ GPTFuzz attack completed successfully"
    echo "📊 Results saved to: $RESULTS_DIR"
    
    # Display basic result summary if available
    if [ -f "$RESULTS_DIR/results_summary.json" ]; then
        echo "📈 Result summary:"
        cat "$RESULTS_DIR/results_summary.json" | head -10
    fi
    
    # Look for GPTFuzz-specific results
    if [ -f "$RESULTS_DIR/gptfuzz_results.json" ]; then
        echo "📊 GPTFuzz specific results:"
        cat "$RESULTS_DIR/gptfuzz_results.json" | head -10
    fi
    
    # Display final GPU memory status
    if command -v nvidia-smi &> /dev/null; then
        echo "💾 Final GPU memory status:"
        nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits
    fi
    
elif [ $EXECUTION_STATUS -eq 124 ]; then
    echo "⏰ GPTFuzz attack timed out after $TIMEOUT_MINUTES minutes"
    exit 124
else
    echo "❌ GPTFuzz attack failed with exit code: $EXECUTION_STATUS"
    echo "📋 Check logs in: $RESULTS_DIR/gptfuzz_execution.log"
    
    # Display GPU memory status on failure to help debug
    if command -v nvidia-smi &> /dev/null; then
        echo "💾 GPU memory status at failure:"
        nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits
    fi
    
    exit $EXECUTION_STATUS
fi

# Display final status
echo "----------------------------------------"
echo "🏁 GPTFuzz attack execution completed"
echo "📊 Model: $MODEL_NAME"
echo "🤖 Mutation model: $MUTATION_MODEL"
echo "🎯 Max queries: $MAX_QUERY"
echo "🔓 Max jailbreaks: $MAX_JAILBREAK"
echo "🔄 Max iterations: $MAX_ITERATION"
echo "📂 Results: $RESULTS_DIR"
echo "🕐 Execution time: $((SECONDS / 60)) minutes and $((SECONDS % 60)) seconds"