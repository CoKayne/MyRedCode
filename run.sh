#!/usr/bin/env bash
#
# RedCode-Gen Experiment Runner - Shell Wrapper
#
# A simplified interface to run experiments with clear output organization.
#
# Usage:
#   ./run.sh                          # Show help
#   ./run.sh --list                   # List available presets
#   ./run.sh --preset qwen-7b-baseline
#   ./run.sh --preset qwen-7b-retrieval --experiment defense_test
#   ./run.sh --batch qwen-7b-baseline qwen-7b-retrieval
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU settings
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║                     RedCode-Gen Experiment Runner                     ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    print_header
    cat << 'EOF'
USAGE:
    ./run.sh [OPTIONS] [ARGUMENTS]

OPTIONS:
    --list, -l              List all available preset configurations
    --preset, -p NAME       Run experiment with preset configuration
    --batch NAME1 NAME2...  Run multiple presets sequentially
    --experiment, -e NAME   Custom experiment name (for organizing outputs)
    --dry-run              Show what would be executed without running
    --gpu ID               GPU device ID to use (disables auto-gpu if set)
    --no-auto-gpu          Disable automatic GPU selection, use GPU 0
    --enable-thinking      Enable model thinking mode (Qwen3 only)
    --no-thinking          Disable model thinking mode
    --reasoning            Enable reasoning prompt (default)
    --no-reasoning         Disable reasoning prompt
    --enable-reflection    Enable reflection generation
    --no-reflection        Disable reflection generation
    --enable-retrieval     Enable retrieval-augmented defense
    --no-retrieval         Disable retrieval-augmented defense
    --skip-judging         Skip judge model evaluation
    --help, -h             Show this help message

PRESET CATEGORIES:
    Qwen 2.5 7B Models:
        qwen-7b-baseline        Qwen 2.5 7B without defense
        qwen-7b-reflection      Qwen 2.5 7B with reflection generation
        qwen-7b-retrieval       Qwen 2.5 7B with retrieval defense
        qwen-7b-full-defense    Qwen 2.5 7B with both reflection + retrieval
        qwen-7b-no-reasoning    Qwen 2.5 7B without reasoning prompt

    Qwen3 8B Models (supports --enable-thinking):
        qwen-8b-baseline        Qwen3 8B baseline (no reasoning, no thinking)
        qwen-8b-thinking        Qwen3 8B with model thinking enabled
        qwen-8b-reasoning       Qwen3 8B with prompt reasoning
        qwen-8b-full-reasoning  Qwen3 8B with prompt reasoning + thinking

    Llama Models:
        llama-8b-baseline       Llama 3.1 8B baseline
        llama-8b-reflection     Llama 3.1 8B with reflection generation
        llama-8b-retrieval      Llama 3.1 8B with retrieval defense
        llama-8b-full-defense   Llama 3.1 8B with reflection + retrieval
        llama-8b-no-reasoning   Llama 3.1 8B without reasoning prompt

    DeepSeek Models:
        deepseek-7b-baseline    DeepSeek Coder 7B baseline
        deepseek-7b-reflection  DeepSeek Coder 7B with reflection generation
        deepseek-7b-retrieval   DeepSeek Coder 7B with retrieval defense
        deepseek-7b-full-defense DeepSeek Coder 7B with reflection + retrieval
        deepseek-7b-no-reasoning DeepSeek Coder 7B without reasoning prompt

    API Models:
        gpt4-baseline           GPT-4 baseline
        gpt4-retrieval          GPT-4 with retrieval defense
        claude-baseline         Claude 3.5 Sonnet baseline
        claude-retrieval        Claude 3.5 Sonnet with retrieval defense

EXAMPLES:
    # List all presets
    ./run.sh --list

    # Run a single preset
    ./run.sh --preset qwen-7b-baseline

    # Run with custom experiment name
    ./run.sh --preset qwen-7b-retrieval --experiment defense_comparison

    # Run multiple presets for comparison
    ./run.sh --batch qwen-7b-baseline qwen-7b-retrieval qwen-7b-full-defense

    # Dry run to see configuration
    ./run.sh --preset qwen-7b-baseline --dry-run

    # Use specific GPU (disables auto-selection)
    ./run.sh --preset llama-8b-baseline --gpu 1

    # Run Qwen3 with thinking enabled
    ./run.sh --preset qwen-8b-thinking

    # Run with retrieval and skip judging
    ./run.sh --preset qwen-7b-retrieval --skip-judging

    # Override preset options
    ./run.sh --preset qwen-7b-baseline --enable-retrieval

OUTPUT STRUCTURE:
    outputs/
    └── {experiment_name}/
        └── {model}_{config}_{timestamp}/
            ├── experiment_config.json   # Full configuration
            ├── solutions/               # Generated code
            ├── logs/                    # Interaction logs
            ├── reasoning/               # Extracted reasoning
            └── results/                 # Evaluation CSVs

EOF
}

list_presets() {
    python run.py --list-presets
}

run_experiment() {
    local args=("$@")
    python run.py "${args[@]}"
}

# Parse arguments
PRESET=""
EXPERIMENT="default"
DRY_RUN=""
GPU=""
BATCH_MODE=false
BATCH_PRESETS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --list|-l)
            list_presets
            exit 0
            ;;
        --preset|-p)
            PRESET="$2"
            shift 2
            ;;
        --experiment|-e)
            EXPERIMENT="$2"
            shift 2
            ;;
        --batch)
            BATCH_MODE=true
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                BATCH_PRESETS+=("$1")
                shift
            done
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --gpu)
            GPU="$2"
            export CUDA_VISIBLE_DEVICES="$GPU"
            EXTRA_ARGS+=("--gpu" "$GPU")
            shift 2
            ;;
        --no-auto-gpu)
            EXTRA_ARGS+=("--no-auto-gpu")
            shift
            ;;
        --enable-thinking)
            EXTRA_ARGS+=("--enable-thinking")
            shift
            ;;
        --no-thinking)
            EXTRA_ARGS+=("--no-thinking")
            shift
            ;;
        --reasoning|--with-reasoning)
            EXTRA_ARGS+=("--reasoning")
            shift
            ;;
        --no-reasoning)
            EXTRA_ARGS+=("--no-reasoning")
            shift
            ;;
        --enable-reflection)
            EXTRA_ARGS+=("--enable-reflection")
            shift
            ;;
        --no-reflection)
            EXTRA_ARGS+=("--no-reflection")
            shift
            ;;
        --enable-retrieval)
            EXTRA_ARGS+=("--enable-retrieval")
            shift
            ;;
        --no-retrieval)
            EXTRA_ARGS+=("--no-retrieval")
            shift
            ;;
        --skip-judging)
            EXTRA_ARGS+=("--skip-judging")
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# Main execution
print_header

if [[ "$BATCH_MODE" == true ]]; then
    if [[ ${#BATCH_PRESETS[@]} -eq 0 ]]; then
        echo -e "${RED}Error: --batch requires at least one preset name${NC}"
        exit 1
    fi
    echo -e "${GREEN}Batch mode: Running ${#BATCH_PRESETS[@]} experiments${NC}"
    echo "Presets: ${BATCH_PRESETS[*]}"
    echo ""
    run_experiment --batch "${BATCH_PRESETS[@]}" --experiment-name "$EXPERIMENT" $DRY_RUN "${EXTRA_ARGS[@]}"
elif [[ -n "$PRESET" ]]; then
    echo -e "${GREEN}Running preset: $PRESET${NC}"
    echo "Experiment: $EXPERIMENT"
    [[ -n "$GPU" ]] && echo "GPU: $GPU"
    echo ""
    run_experiment --preset "$PRESET" --experiment-name "$EXPERIMENT" $DRY_RUN "${EXTRA_ARGS[@]}"
else
    print_help
    exit 0
fi
