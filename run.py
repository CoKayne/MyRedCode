#!/usr/bin/env python3
"""
RedCode-Gen Experiment Runner

A clear interface to run experiments with different models and parameters,
with automatic output organization to prevent confusion between results.

Usage:
    # List available presets
    python run.py --list-presets

    # Run with a preset configuration
    python run.py --preset qwen-7b-baseline

    # Run with custom parameters
    python run.py --model Qwen/Qwen2.5-7B-Instruct --experiment-name my_experiment

    # Run with retrieval defense
    python run.py --preset qwen-7b-retrieval --retrieval-similarity 0.6

    # Compare multiple configurations
    python run.py --compare qwen-7b-baseline qwen-7b-retrieval
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


# ============================================================================
# GPU Auto-Selection
# ============================================================================

def select_best_gpu(verbose: bool = True) -> Optional[str]:
    """
    Automatically select the GPU with the most free memory using GPUtil.

    Args:
        verbose: Print GPU selection info.

    Returns:
        GPU ID as string (e.g., "0", "1"), or None if no GPUs available or GPUtil not installed.
    """
    try:
        from GPUtil import getGPUs
        gpus = getGPUs()

        if not gpus:
            if verbose:
                print("No GPUs detected, will use CPU instead")
            return None

        if verbose:
            print("\nGPU Memory Status:")
            print("-" * 60)

        best = max(gpus, key=lambda g: g.memoryFree)

        if verbose:
            for gpu in gpus:
                used = gpu.memoryTotal - gpu.memoryFree
                usage_pct = (used / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0
                marker = " <-- selected" if gpu.id == best.id else ""
                print(f"  GPU {gpu.id}: {gpu.memoryFree:.0f} MB free / {gpu.memoryTotal:.0f} MB total ({usage_pct:.1f}% used){marker}")
            print("-" * 60)

        return str(best.id)

    except ImportError:
        if verbose:
            print("Warning: GPUtil not installed. Install with: pip install gputil")
            print("         Using default GPU setting.")
        return None
    except Exception as e:
        if verbose:
            print(f"Warning: Could not auto-select GPU: {e}")
        return None


def check_judge_api_key(judge_model: str) -> bool:
    """
    Check if the required API key for the judge model is available.

    Returns:
        True if API key is available, False otherwise.
    """
    if judge_model.startswith("gpt"):
        return bool(os.environ.get("OPENAI_API_KEY"))
    elif judge_model.startswith("claude"):
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    elif judge_model.startswith("gemini") or judge_model.startswith("models/"):
        return bool(os.environ.get("GEMINI_API_KEY"))
    # For other models (e.g., local transformers), assume available
    return True


# ============================================================================
# Preset Configurations
# ============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # Qwen 2.5 7B Models (No thinking support)
    # =========================================================================
    "qwen-7b-baseline": {
        "description": "Qwen 2.5 7B baseline without defense",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "qwen-7b-reflection": {
        "description": "Qwen 2.5 7B with reflection generation (builds defense DB)",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": True,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "qwen-7b-retrieval": {
        "description": "Qwen 2.5 7B with retrieval-augmented defense",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "qwen-7b-full-defense": {
        "description": "Qwen 2.5 7B with both reflection and retrieval",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": True,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "qwen-7b-no-reasoning": {
        "description": "Qwen 2.5 7B without reasoning prompt",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "qwen-7b-no-reasoning-reflection": {
        "description": "Qwen 2.5 7B without reasoning but with reflection",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": True,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "qwen-7b-no-reasoning-retrieval": {
        "description": "Qwen 2.5 7B without reasoning but with retrieval defense",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": False,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "qwen-7b-no-reasoning-full-defense": {
        "description": "Qwen 2.5 7B without reasoning but with reflection and retrieval",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": True,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },

    # =========================================================================
    # Qwen3 8B Models (Supports enable_thinking)
    # =========================================================================
    "qwen-8b-baseline": {
        "description": "Qwen3 8B baseline (no prompt reasoning, no thinking)",
        "model": "Qwen/Qwen3-8B",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "qwen-8b-thinking": {
        "description": "Qwen3 8B with model thinking enabled",
        "model": "Qwen/Qwen3-8B",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": True,
    },
    "qwen-8b-reasoning": {
        "description": "Qwen3 8B with prompt reasoning (asks for <think> block)",
        "model": "Qwen/Qwen3-8B",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "qwen-8b-full-reasoning": {
        "description": "Qwen3 8B with prompt reasoning + model thinking",
        "model": "Qwen/Qwen3-8B",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": True,
    },

    # =========================================================================
    # Llama 3.1 8B Models
    # =========================================================================
    "llama-8b-baseline": {
        "description": "Llama 3.1 8B baseline",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "llama-8b-reflection": {
        "description": "Llama 3.1 8B with reflection generation (builds defense DB)",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": True,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "llama-8b-retrieval": {
        "description": "Llama 3.1 8B with retrieval-augmented defense",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "llama-8b-full-defense": {
        "description": "Llama 3.1 8B with both reflection and retrieval",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": True,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "llama-8b-no-reasoning": {
        "description": "Llama 3.1 8B without reasoning prompt",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "llama-8b-no-reasoning-reflection": {
        "description": "Llama 3.1 8B without reasoning but with reflection",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": True,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "llama-8b-no-reasoning-retrieval": {
        "description": "Llama 3.1 8B without reasoning but with retrieval defense",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": False,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "llama-8b-no-reasoning-full-defense": {
        "description": "Llama 3.1 8B without reasoning but with reflection and retrieval",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": True,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },

    # =========================================================================
    # DeepSeek Coder 7B Models
    # =========================================================================
    "deepseek-7b-baseline": {
        "description": "DeepSeek Coder 7B baseline",
        "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "deepseek-7b-reflection": {
        "description": "DeepSeek Coder 7B with reflection generation",
        "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": True,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "deepseek-7b-retrieval": {
        "description": "DeepSeek Coder 7B with retrieval defense",
        "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "deepseek-7b-full-defense": {
        "description": "DeepSeek Coder 7B with both reflection and retrieval",
        "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "use_transformers": True,
        "prompt_with_reasoning": True,
        "enable_reflection": True,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "deepseek-7b-no-reasoning": {
        "description": "DeepSeek Coder 7B without reasoning prompt",
        "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "deepseek-7b-no-reasoning-reflection": {
        "description": "DeepSeek Coder 7B without reasoning but with reflection",
        "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": True,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "deepseek-7b-no-reasoning-retrieval": {
        "description": "DeepSeek Coder 7B without reasoning but with retrieval defense",
        "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": False,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "deepseek-7b-no-reasoning-full-defense": {
        "description": "DeepSeek Coder 7B without reasoning but with reflection and retrieval",
        "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "use_transformers": True,
        "prompt_with_reasoning": False,
        "enable_reflection": True,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },

    # =========================================================================
    # API Models (GPT-4, Claude) - Minimal presets
    # =========================================================================
    "gpt4-baseline": {
        "description": "GPT-4 baseline (API)",
        "model": "gpt-4",
        "use_transformers": False,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "gpt4-retrieval": {
        "description": "GPT-4 with retrieval defense (API)",
        "model": "gpt-4",
        "use_transformers": False,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
    "claude-baseline": {
        "description": "Claude 3.5 Sonnet baseline (API)",
        "model": "claude-3-5-sonnet-20241022",
        "use_transformers": False,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": False,
        "enable_thinking": False,
    },
    "claude-retrieval": {
        "description": "Claude 3.5 Sonnet with retrieval defense (API)",
        "model": "claude-3-5-sonnet-20241022",
        "use_transformers": False,
        "prompt_with_reasoning": True,
        "enable_reflection": False,
        "enable_retrieval": True,
        "retrieval_similarity_threshold": 0.5,
        "retrieval_max_results": 3,
        "enable_thinking": False,
    },
}


# ============================================================================
# Output Directory Management
# ============================================================================

def get_experiment_dir(base_output: str, experiment_name: str, model: str, config: Dict[str, Any]) -> Path:
    """
    Generate a unique, descriptive output directory for an experiment.

    Structure: outputs/{experiment_name}/{model_short}_{config_hash}_{timestamp}/
    """
    # Create short model name
    model_short = model.split("/")[-1].replace("-", "_").lower()
    if len(model_short) > 30:
        model_short = model_short[:30]

    # Create config descriptor
    config_parts = []

    # Reasoning mode indicator
    if config.get("prompt_with_reasoning"):
        config_parts.append("reasoning")
    else:
        config_parts.append("no_reasoning")

    # Thinking mode indicator (only relevant for Qwen3)
    if config.get("enable_thinking"):
        config_parts.append("thinking")

    # Defense mechanisms
    if config.get("enable_retrieval"):
        config_parts.append("retrieval")
    if config.get("enable_reflection"):
        config_parts.append("reflection")

    if not config.get("enable_retrieval") and not config.get("enable_reflection"):
        config_parts.append("baseline")

    config_str = "_".join(config_parts)

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build directory name
    dir_name = f"{model_short}_{config_str}_{timestamp}"

    return Path(base_output) / experiment_name / dir_name


def setup_experiment_dirs(experiment_dir: Path) -> Dict[str, Path]:
    """Create and return all necessary subdirectories for an experiment."""
    dirs = {
        "root": experiment_dir,
        "solutions": experiment_dir / "solutions",
        "logs": experiment_dir / "logs",
        "reasoning": experiment_dir / "reasoning",
        "results": experiment_dir / "results",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def save_experiment_config(experiment_dir: Path, config: Dict[str, Any], args: argparse.Namespace):
    """Save the full experiment configuration for reproducibility."""
    config_file = experiment_dir / "experiment_config.json"

    full_config = {
        "timestamp": datetime.now().isoformat(),
        "command_line_args": vars(args),
        "resolved_config": config,
        "preset_used": args.preset if hasattr(args, 'preset') else None,
    }

    with open(config_file, 'w') as f:
        json.dump(full_config, f, indent=2, default=str)

    print(f"Configuration saved to: {config_file}")


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RedCode-Gen Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available presets
  python run.py --list-presets

  # Run a preset configuration
  python run.py --preset qwen-7b-baseline

  # Run with custom model
  python run.py --model Qwen/Qwen2.5-7B-Instruct --experiment-name custom_test

  # Run with retrieval defense and custom threshold
  python run.py --preset qwen-7b-retrieval --retrieval-similarity 0.6

  # Run multiple presets for comparison
  python run.py --batch qwen-7b-baseline qwen-7b-retrieval

  # Dry run to see what would be executed
  python run.py --preset qwen-7b-baseline --dry-run
        """
    )

    # Preset and experiment naming
    parser.add_argument("--preset", "-p", type=str, choices=list(PRESETS.keys()),
                        help="Use a preset configuration")
    parser.add_argument("--list-presets", action="store_true",
                        help="List all available preset configurations")
    parser.add_argument("--experiment-name", "-e", type=str, default="default",
                        help="Name for this experiment (used in output path)")
    parser.add_argument("--batch", nargs="+", type=str, metavar="PRESET",
                        help="Run multiple presets in batch mode")

    # Model configuration
    parser.add_argument("--model", "-m", type=str,
                        help="Model to evaluate (overrides preset)")
    parser.add_argument("--judge-model", type=str, default="gpt-4",
                        help="Model for judging generated code")
    parser.add_argument("--use-transformers", action="store_true", default=None,
                        help="Use local HuggingFace Transformers")
    parser.add_argument("--no-transformers", action="store_true",
                        help="Use API-based models")

    # Evaluation settings
    parser.add_argument("--evaluation-type", choices=["normal", "agent"], default="normal",
                        help="Type of evaluation")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of rounds for agent evaluation")
    parser.add_argument("--skip-judging", action="store_true",
                        help="Skip judge model evaluation")
    parser.add_argument("--use-virustotal", action="store_true",
                        help="Enable VirusTotal evaluation")

    # Reasoning settings
    parser.add_argument("--reasoning", "--with-reasoning", action="store_true", default=None,
                        dest="prompt_with_reasoning",
                        help="Use reasoning prompt (default)")
    parser.add_argument("--no-reasoning", action="store_false", dest="prompt_with_reasoning",
                        help="Disable reasoning prompt")

    # Thinking settings (for Qwen3 models)
    parser.add_argument("--enable-thinking", action="store_true", default=None,
                        help="Enable model thinking mode (Qwen3 only)")
    parser.add_argument("--no-thinking", action="store_false", dest="enable_thinking",
                        help="Disable model thinking mode")

    # Reflection settings
    parser.add_argument("--enable-reflection", action="store_true", default=None,
                        help="Enable reflection generation")
    parser.add_argument("--no-reflection", action="store_false", dest="enable_reflection",
                        help="Disable reflection generation")
    parser.add_argument("--reflection-db", type=str, default="./reflection_db",
                        help="Path to reflection database")
    parser.add_argument("--reflection-threshold", type=int, default=5,
                        help="Minimum score to generate reflection")

    # Retrieval settings
    parser.add_argument("--enable-retrieval", action="store_true", default=None,
                        help="Enable retrieval-augmented defense")
    parser.add_argument("--no-retrieval", action="store_false", dest="enable_retrieval",
                        help="Disable retrieval-augmented defense")
    parser.add_argument("--retrieval-similarity", type=float, default=0.5,
                        help="Similarity threshold for retrieval (0.0-1.0)")
    parser.add_argument("--retrieval-max-results", type=int, default=3,
                        help="Maximum reflections to retrieve")
    parser.add_argument("--retrieval-score-threshold", type=int, default=5,
                        help="Minimum score of stored reflections to use")

    # Paths
    parser.add_argument("--base-dir", type=str, default="./dataset/RedCode-Gen",
                        help="Directory containing test prompts")
    parser.add_argument("--output-base", type=str, default="./outputs",
                        help="Base directory for all outputs")

    # Hardware settings
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID(s) to use (disables auto-gpu if set)")
    parser.add_argument("--no-auto-gpu", action="store_true",
                        help="Disable automatic GPU selection, use GPU 0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type for transformers models")

    # Execution control
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration without running")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    return parser.parse_args()


# ============================================================================
# Configuration Resolution
# ============================================================================

def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve final configuration from preset and command-line overrides."""

    # Start with preset if specified
    if args.preset:
        config = PRESETS[args.preset].copy()
    else:
        config = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "use_transformers": True,
            "prompt_with_reasoning": True,
            "enable_reflection": False,
            "enable_retrieval": False,
        }

    # Apply command-line overrides
    if args.model:
        config["model"] = args.model

    if args.use_transformers is not None:
        config["use_transformers"] = args.use_transformers
    if args.no_transformers:
        config["use_transformers"] = False

    if args.prompt_with_reasoning is not None:
        config["prompt_with_reasoning"] = args.prompt_with_reasoning

    # Handle enable_thinking
    if args.enable_thinking is not None:
        config["enable_thinking"] = args.enable_thinking
    # If not explicitly set, use preset default or False
    if "enable_thinking" not in config:
        config["enable_thinking"] = False

    if args.enable_reflection is not None:
        config["enable_reflection"] = args.enable_reflection

    if args.enable_retrieval is not None:
        config["enable_retrieval"] = args.enable_retrieval

    # Set retrieval parameters if retrieval is enabled
    if config.get("enable_retrieval"):
        config["retrieval_similarity_threshold"] = args.retrieval_similarity
        config["retrieval_max_results"] = args.retrieval_max_results
        config["retrieval_score_threshold"] = args.retrieval_score_threshold

    # Other settings
    config["judge_model"] = args.judge_model
    config["evaluation_type"] = args.evaluation_type
    config["rounds"] = args.rounds
    config["use_virustotal"] = args.use_virustotal

    # Auto-skip judging if API key for judge model is not available
    if args.skip_judging:
        config["skip_judging"] = True
    elif not check_judge_api_key(args.judge_model):
        print(f"\nWarning: API key for judge model '{args.judge_model}' not found.")
        print("         Automatically enabling --skip-judging to generate code without evaluation.")
        print("         To enable judging, set the appropriate API key environment variable:")
        print("           - GPT models: OPENAI_API_KEY")
        print("           - Claude models: ANTHROPIC_API_KEY")
        print("           - Gemini models: GEMINI_API_KEY")
        config["skip_judging"] = True
    else:
        config["skip_judging"] = False
    config["base_dir"] = args.base_dir
    config["reflection_db_path"] = args.reflection_db
    config["reflection_score_threshold"] = args.reflection_threshold
    config["dtype"] = args.dtype

    # GPU selection: auto-gpu is default unless --gpu is specified or --no-auto-gpu is set
    if args.gpu is not None:
        # Explicit GPU specified - disable auto-gpu
        config["auto_gpu"] = False
        config["gpu"] = args.gpu
    elif args.no_auto_gpu:
        # Explicitly disabled auto-gpu - use GPU 0
        config["auto_gpu"] = False
        config["gpu"] = "0"
    else:
        # Default: auto-gpu enabled
        config["auto_gpu"] = True
        config["gpu"] = None  # Will be set at runtime

    return config


# ============================================================================
# Execution
# ============================================================================

def build_command(config: Dict[str, Any], dirs: Dict[str, Path]) -> List[str]:
    """Build the command to run the evaluation."""
    cmd = [
        sys.executable,
        "evaluation/RedCode_Gen/main.py",
        "--model", config["model"],
        "--judge_model", config["judge_model"],
        "--base_dir", config["base_dir"],
        "--output_dir", str(dirs["solutions"]),
        "--logs_dir", str(dirs["logs"]),
        "--reasoning_dir", str(dirs["reasoning"]),
        "--evaluation_type", config["evaluation_type"],
        "--rounds", str(config["rounds"]),
        "--prompt_with_reasoning", "1" if config.get("prompt_with_reasoning", True) else "0",
        "--reflection_db_path", config["reflection_db_path"],
        "--reflection_score_threshold", str(config["reflection_score_threshold"]),
        "--transformers_dtype", config["dtype"],
    ]

    if config.get("use_transformers"):
        cmd.append("--use_transformers")

    if config.get("skip_judging"):
        cmd.append("--skip_judging")

    if config.get("use_virustotal"):
        cmd.append("--use_virustotal")

    if config.get("enable_reflection"):
        cmd.append("--enable_reflection")

    if config.get("enable_thinking"):
        cmd.append("--enable_thinking")

    if config.get("enable_retrieval"):
        cmd.extend([
            "--enable_retrieval",
            "--retrieval_similarity_threshold", str(config.get("retrieval_similarity_threshold", 0.5)),
            "--retrieval_max_results", str(config.get("retrieval_max_results", 3)),
            "--retrieval_score_threshold", str(config.get("retrieval_score_threshold", 5)),
        ])

    return cmd


def run_experiment(args: argparse.Namespace, config: Dict[str, Any], preset_name: Optional[str] = None) -> bool:
    """Run a single experiment with the given configuration."""

    # Determine experiment name
    exp_name = args.experiment_name
    if preset_name:
        exp_name = f"{exp_name}/{preset_name}" if exp_name != "default" else preset_name

    # Setup directories
    experiment_dir = get_experiment_dir(args.output_base, exp_name, config["model"], config)
    dirs = setup_experiment_dirs(experiment_dir)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70)
    print(f"Model: {config['model']}")
    print(f"Output: {experiment_dir}")
    print(f"Reasoning: {'enabled' if config.get('prompt_with_reasoning') else 'disabled'}")
    print(f"Reflection: {'enabled' if config.get('enable_reflection') else 'disabled'}")
    print(f"Retrieval: {'enabled' if config.get('enable_retrieval') else 'disabled'}")
    if config.get("enable_retrieval"):
        print(f"  - Similarity threshold: {config.get('retrieval_similarity_threshold', 0.5)}")
        print(f"  - Max results: {config.get('retrieval_max_results', 3)}")
    print("=" * 70)

    # Save configuration
    save_experiment_config(experiment_dir, config, args)

    # Build command
    cmd = build_command(config, dirs)

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        print(" ".join(cmd))
        return True

    # Set environment
    env = os.environ.copy()

    # GPU selection: auto-gpu or manual
    if config.get("auto_gpu"):
        selected_gpu = select_best_gpu(verbose=True)
        if selected_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = selected_gpu
            config["gpu"] = selected_gpu  # Update config for logging
        # If None, don't set CUDA_VISIBLE_DEVICES - let main.py handle it or use all GPUs
    else:
        env["CUDA_VISIBLE_DEVICES"] = config["gpu"]
        print(f"\nUsing GPU: {config['gpu']}")

    # Run the experiment
    print(f"\nStarting experiment at {datetime.now().isoformat()}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent)

        # Copy evaluation results to experiment directory
        model_safe = config["model"].replace("/", "_")
        results_file = f"evaluation_results_{model_safe}.csv"
        if Path(results_file).exists():
            import shutil
            shutil.move(results_file, dirs["results"] / results_file)
            print(f"\nResults saved to: {dirs['results'] / results_file}")

        success = result.returncode == 0
        status = "SUCCESS" if success else f"FAILED (exit code {result.returncode})"
        print(f"\nExperiment completed: {status}")

        # Save completion status
        status_file = experiment_dir / "status.json"
        with open(status_file, 'w') as f:
            json.dump({
                "completed": datetime.now().isoformat(),
                "success": success,
                "exit_code": result.returncode,
            }, f, indent=2)

        return success

    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        return False


def list_presets():
    """Print all available presets with descriptions."""
    print("\n" + "=" * 70)
    print("AVAILABLE PRESETS")
    print("=" * 70)

    # Group presets by model family
    groups = {}
    for name, config in PRESETS.items():
        model = config["model"]
        if "qwen" in model.lower():
            group = "Qwen Models"
        elif "llama" in model.lower():
            group = "Llama Models"
        elif "deepseek" in model.lower():
            group = "DeepSeek Models"
        elif "gpt" in model.lower():
            group = "OpenAI API Models"
        elif "claude" in model.lower():
            group = "Anthropic API Models"
        else:
            group = "Other Models"

        if group not in groups:
            groups[group] = []
        groups[group].append((name, config))

    for group_name, presets in groups.items():
        print(f"\n{group_name}:")
        print("-" * 40)
        for name, config in presets:
            desc = config.get("description", "No description")
            features = []
            if config.get("enable_retrieval"):
                features.append("retrieval")
            if config.get("enable_reflection"):
                features.append("reflection")
            if config.get("prompt_with_reasoning") is False:
                features.append("no-reasoning")
            if config.get("enable_thinking"):
                features.append("thinking")

            feature_str = f" [{', '.join(features)}]" if features else ""
            print(f"  {name:25s} {desc}{feature_str}")

    print("\n" + "=" * 70)
    print("Usage: python run.py --preset <preset_name>")
    print("=" * 70 + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    args = parse_args()

    # Handle special commands
    if args.list_presets:
        list_presets()
        return 0

    # Batch mode
    if args.batch:
        print(f"\nBatch mode: Running {len(args.batch)} experiments")
        results = []
        for preset_name in args.batch:
            if preset_name not in PRESETS:
                print(f"Warning: Unknown preset '{preset_name}', skipping")
                continue

            args.preset = preset_name
            config = resolve_config(args)
            success = run_experiment(args, config, preset_name)
            results.append((preset_name, success))

        # Summary
        print("\n" + "=" * 70)
        print("BATCH SUMMARY")
        print("=" * 70)
        for preset_name, success in results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  {preset_name:30s} {status}")

        failed = sum(1 for _, s in results if not s)
        if failed:
            print(f"\n{failed}/{len(results)} experiments failed")
            return 1
        return 0

    # Single experiment mode
    if not args.preset and not args.model:
        print("Error: Please specify --preset or --model")
        print("Use --list-presets to see available configurations")
        return 1

    config = resolve_config(args)
    success = run_experiment(args, config)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
