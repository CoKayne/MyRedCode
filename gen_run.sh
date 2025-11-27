#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-1.5-pro}"
BASE_DIR="${BASE_DIR:-dataset/RedCode-Gen}"
OUTPUT_DIR="${OUTPUT_DIR:-solutions_Llama_reasoning}"
LOGS_DIR="${LOGS_DIR:-logs_Llama_reasoning}"
REASONING_DIR="${REASONING_DIR:-reasoning_Llama_reasoning}"
EVAL_TYPE="${EVALUATION_TYPE:-normal}"
ROUNDS="${ROUNDS:-5}"
PROMPT_WITH_REASONING="${PROMPT_WITH_REASONING:-1}"
SKIP_JUDGING=1 # Default to skipping judging to save costs

ARGS=(
  --model "$MODEL"
  --judge_model "$JUDGE_MODEL"
  --base_dir "$BASE_DIR"
  --output_dir "$OUTPUT_DIR"
  --logs_dir "$LOGS_DIR"
  --reasoning_dir "$REASONING_DIR"
  --evaluation_type "$EVAL_TYPE"
  --rounds "$ROUNDS"
  --prompt_with_reasoning "$PROMPT_WITH_REASONING"
)

if [[ "${USE_TRANSFORMERS:-1}" =~ ^(1|true|yes|on)$ ]]; then
  ARGS+=(--use_transformers)
fi

if [[ "${SKIP_JUDGING:-0}" =~ ^(1|true|yes|on)$ ]]; then
  ARGS+=(--skip_judging)
fi

if [[ "${USE_VIRUSTOTAL:-0}" =~ ^(1|true|yes|on)$ ]]; then
  ARGS+=(--use_virustotal)
fi

python evaluation/RedCode_Gen/main.py "${ARGS[@]}" "$@"
