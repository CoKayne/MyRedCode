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
    --gpu ID               GPU device ID to use (default: 0)
    --help, -h             Show this help message

PRESET CATEGORIES:
    Qwen Models:
        qwen-7b-baseline        Qwen 2.5 7B without defense
        qwen-7b-reflection      Qwen 2.5 7B with reflection generation
        qwen-7b-retrieval       Qwen 2.5 7B with retrieval defense
        qwen-7b-full-defense    Qwen 2.5 7B with both reflection + retrieval
        qwen-7b-no-reasoning    Qwen 2.5 7B without reasoning prompt

    Llama Models:
        llama-8b-baseline       Llama 3.1 8B baseline
        llama-8b-retrieval      Llama 3.1 8B with retrieval defense

    DeepSeek Models:
        deepseek-7b-baseline    DeepSeek Coder 7B baseline
        deepseek-7b-retrieval   DeepSeek Coder 7B with retrieval

    API Models:
        gpt4-baseline           GPT-4 baseline
        gpt4-retrieval          GPT-4 with retrieval defense
        claude-baseline         Claude 3.5 Sonnet baseline

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

    # Use specific GPU
    ./run.sh --preset llama-8b-baseline --gpu 1

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
GPU="0"
BATCH_MODE=false
BATCH_PRESETS=()

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
            shift 2
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
    run_experiment --batch "${BATCH_PRESETS[@]}" --experiment-name "$EXPERIMENT" $DRY_RUN
elif [[ -n "$PRESET" ]]; then
    echo -e "${GREEN}Running preset: $PRESET${NC}"
    echo "Experiment: $EXPERIMENT"
    echo "GPU: $GPU"
    echo ""
    run_experiment --preset "$PRESET" --experiment-name "$EXPERIMENT" --gpu "$GPU" $DRY_RUN
else
    print_help
    exit 0
fi
