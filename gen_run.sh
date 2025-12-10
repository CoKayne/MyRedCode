#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-1.5-pro}"
BASE_DIR="${BASE_DIR:-dataset/RedCode-Gen}"
OUTPUT_DIR="${OUTPUT_DIR:-solutions_Qwen2.5_reasoning}"
LOGS_DIR="${LOGS_DIR:-logs_Qwen2.5_reasoning}"
REASONING_DIR="${REASONING_DIR:-reasoning_Qwen2.5_reasoning}"
EVAL_TYPE="${EVALUATION_TYPE:-normal}"
ROUNDS="${ROUNDS:-5}"
PROMPT_WITH_REASONING="${PROMPT_WITH_REASONING:-1}"
SKIP_JUDGING=1 # Default to skipping judging to save costs
ENABLE_REFLECTION=1
ENABLE_RETRIEVAL=0 # Set to 1 to enable retrieval-augmented defense
RETRIEVAL_SIMILARITY_THRESHOLD="${RETRIEVAL_SIMILARITY_THRESHOLD:-0.5}"
RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-3}"
RETRIEVAL_SCORE_THRESHOLD="${RETRIEVAL_SCORE_THRESHOLD:-5}"

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

if [[ "${ENABLE_REFLECTION:-0}" =~ ^(1|true|yes|on)$ ]]; then
  ARGS+=(--enable_reflection)
fi

if [[ "${ENABLE_RETRIEVAL:-0}" =~ ^(1|true|yes|on)$ ]]; then
  ARGS+=(--enable_retrieval)
  ARGS+=(--retrieval_similarity_threshold "$RETRIEVAL_SIMILARITY_THRESHOLD")
  ARGS+=(--retrieval_max_results "$RETRIEVAL_MAX_RESULTS")
  ARGS+=(--retrieval_score_threshold "$RETRIEVAL_SCORE_THRESHOLD")
fi

python evaluation/RedCode_Gen/main.py "${ARGS[@]}" "$@"
