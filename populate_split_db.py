#!/usr/bin/env python3
"""
Populate reflection database with training split only.

This script populates a reflection database from existing evaluation results,
filtering to only include the training split (indices 1-10).

Usage:
    python populate_split_db.py \
        --source-dir ./RESULTS/llama_reasoning_results \
        --db-path ./reflection_db_split/llama_reasoning \
        --source-model-name llama_reasoning
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add evaluation/RedCode_Gen to path
sys.path.insert(0, str(Path(__file__).parent / "evaluation" / "RedCode_Gen"))

from clients import get_client
from reflection_utils import generate_reflection, parse_reflection_response
from reflection_db import ReflectionDatabase
from dataset_split import is_train_file, is_test_file, get_file_index, TRAIN_INDICES, TEST_INDICES


def detect_results_structure(results_dir: Path) -> Dict[str, Path]:
    """
    Auto-detect the structure of a results folder.

    Returns:
        dict with keys: logs_dir, solutions_dir, score_file, reasoning_dir (optional)
    """
    structure = {
        "logs_dir": None,
        "solutions_dir": None,
        "score_file": None,
        "reasoning_dir": None
    }

    # Find logs folder
    for pattern in ["logs_*", "logs", "*logs*"]:
        for path in results_dir.glob(pattern):
            if path.is_dir():
                structure["logs_dir"] = path
                break
        if structure["logs_dir"]:
            break

    # Find solutions folder
    for pattern in ["solutions_*", "solutions", "*solutions*"]:
        for path in results_dir.glob(pattern):
            if path.is_dir():
                structure["solutions_dir"] = path
                break
        if structure["solutions_dir"]:
            break

    # Find score file
    score_file = results_dir / "score"
    if score_file.exists():
        structure["score_file"] = score_file

    # Find reasoning folder (optional)
    for pattern in ["reasoning_*", "reasoning", "*reasoning*"]:
        for path in results_dir.glob(pattern):
            if path.is_dir():
                structure["reasoning_dir"] = path
                break
        if structure["reasoning_dir"]:
            break

    return structure


def parse_log_file(log_path: Path, has_reasoning: bool = False) -> Dict[str, str]:
    """
    Parse a log file to extract relevant information.

    Args:
        log_path: Path to the log file
        has_reasoning: Whether to expect reasoning blocks

    Returns:
        dict with keys: family, file_name, user_request, model_response, extracted_code, reasoning
    """
    log_text = log_path.read_text(encoding='utf-8', errors='ignore')

    result = {
        "family": "",
        "file_name": "",
        "user_request": "",
        "model_response": "",
        "extracted_code": "",
        "reasoning": ""
    }

    # Extract Family
    family_match = re.search(r'^Family:\s*(.+)$', log_text, re.MULTILINE)
    if family_match:
        result["family"] = family_match.group(1).strip()

    # Extract File
    file_match = re.search(r'^File:\s*(.+)$', log_text, re.MULTILINE)
    if file_match:
        result["file_name"] = file_match.group(1).strip()

    # Extract User Request
    gen_request_start = log_text.find("=== Generation Request ===")
    model_response_start = log_text.find("=== Model Response ===")

    if gen_request_start != -1 and model_response_start != -1:
        user_request_section = log_text[gen_request_start:model_response_start]
        user_request_match = re.search(r'User Request:\s*(.+)$', user_request_section, re.DOTALL)
        if user_request_match:
            result["user_request"] = user_request_match.group(1).strip()

    # Extract Model Response
    if model_response_start != -1:
        model_response_section = log_text[model_response_start:]
        extracted_code_start = model_response_section.find("=== Extracted Code ===")
        judge_prompt_start = model_response_section.find("=== Judge Prompt ===")

        end_pos = extracted_code_start if extracted_code_start != -1 else judge_prompt_start
        if end_pos != -1:
            result["model_response"] = model_response_section[:end_pos].strip()
        else:
            result["model_response"] = model_response_section.strip()

    # Extract Extracted Code
    extracted_code_start = log_text.find("=== Extracted Code ===")
    if extracted_code_start != -1:
        extracted_code_section = log_text[extracted_code_start:]
        judge_prompt_start = extracted_code_section.find("=== Judge Prompt ===")
        if judge_prompt_start != -1:
            extracted_code_text = extracted_code_section[len("=== Extracted Code ==="):judge_prompt_start].strip()
            result["extracted_code"] = extracted_code_text

    # Extract Reasoning
    if has_reasoning:
        reasoning_match = re.search(r'<think>(.*?)</think>', log_text, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            if result["model_response"]:
                open_tag_pos = result["model_response"].find("<think>")
                if open_tag_pos != -1:
                    remaining = result["model_response"][open_tag_pos + len("<think>"):]
                    code_block_pos = remaining.find("```")
                    close_tag_pos = remaining.find("</think>")

                    if close_tag_pos != -1:
                        result["reasoning"] = remaining[:close_tag_pos].strip()
                    elif code_block_pos != -1:
                        result["reasoning"] = remaining[:code_block_pos].strip()
                    else:
                        result["reasoning"] = remaining.strip()
    else:
        result["reasoning"] = "N/A"

    return result


def parse_score_file(score_path: Path) -> Dict[str, List[int]]:
    """
    Parse a score file to extract scores by family.

    Returns:
        dict mapping family name to list of scores
    """
    score_text = score_path.read_text(encoding='utf-8')
    scores_by_family = {}

    pattern = r'^(\w+):\s*\[(.*?)\]$'
    for line in score_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('---'):
            continue

        match = re.match(pattern, line)
        if match:
            family = match.group(1)
            scores_str = match.group(2)
            scores = [int(s.strip()) for s in scores_str.split(',') if s.strip()]
            scores_by_family[family] = scores

    return scores_by_family


def main():
    parser = argparse.ArgumentParser(
        description="Populate reflection database with training split only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Populate with Llama reflections
    python populate_split_db.py \\
        --source-dir ./RESULTS/llama_reasoning_results \\
        --db-path ./reflection_db_split/llama_reasoning \\
        --source-model-name llama_reasoning

    # Populate with Qwen reflections
    python populate_split_db.py \\
        --source-dir ./RESULTS/qwen_reasoning_results \\
        --db-path ./reflection_db_split/qwen_reasoning \\
        --source-model-name qwen_reasoning
        """
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to the results directory (e.g., ./RESULTS/llama_reasoning_results)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to the output reflection database (e.g., ./reflection_db_split/llama_reasoning)"
    )
    parser.add_argument(
        "--source-model-name",
        type=str,
        required=True,
        help="Name to identify the source model (e.g., llama_reasoning, qwen_reasoning)"
    )
    parser.add_argument(
        "--split-mode",
        choices=["train", "test", "all"],
        default="train",
        help="Which split to populate: train (indices 1-10), test (indices 11-20), or all"
    )
    parser.add_argument(
        "--has-reasoning",
        action="store_true",
        help="Results contain reasoning (will extract from <think> tags)"
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Results do not contain reasoning"
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default="gpt-4o",
        help="Model to use for generating reflections (default: gpt-4o)"
    )
    parser.add_argument(
        "--score-threshold",
        type=int,
        default=0,
        help="Only generate reflections for scores >= threshold (default: 0)"
    )
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        help="Use local transformers model instead of API"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually doing it"
    )

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return 1

    # Detect folder structure
    print(f"Detecting results structure: {source_dir}")
    structure = detect_results_structure(source_dir)

    logs_dir = structure["logs_dir"]
    solutions_dir = structure["solutions_dir"]
    score_file = structure["score_file"]

    if not logs_dir:
        print(f"Error: Cannot find logs folder in {source_dir}")
        return 1
    if not solutions_dir:
        print(f"Error: Cannot find solutions folder in {source_dir}")
        return 1
    if not score_file:
        print(f"Error: Cannot find score file in {source_dir}")
        return 1

    print(f"  Logs folder: {logs_dir}")
    print(f"  Solutions folder: {solutions_dir}")
    print(f"  Score file: {score_file}")

    # Auto-detect reasoning mode
    has_reasoning = None
    if args.has_reasoning:
        has_reasoning = True
    elif args.no_reasoning:
        has_reasoning = False
    else:
        if "no_reasoning" in source_dir.name.lower():
            has_reasoning = False
        elif "reasoning" in source_dir.name.lower():
            has_reasoning = True
        else:
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                sample_log = log_files[0].read_text(encoding='utf-8', errors='ignore')
                has_reasoning = "<think>" in sample_log

    print(f"  Reasoning mode: {'enabled' if has_reasoning else 'disabled'}")
    print(f"  Split mode: {args.split_mode} (indices {list(TRAIN_INDICES) if args.split_mode == 'train' else list(TEST_INDICES) if args.split_mode == 'test' else '1-20'})")

    # Parse score file
    print("\nParsing score file...")
    scores_by_family = parse_score_file(score_file)
    print(f"Found scores for {len(scores_by_family)} families")

    if args.dry_run:
        print("\n[DRY RUN] Would process the following files:")

    # Initialize reflection client and database
    if not args.dry_run:
        print(f"\nInitializing reflection client (model: {args.reflection_model})...")
        config = {
            "model": args.reflection_model,
            "use_transformers": args.use_transformers,
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        }
        try:
            reflection_client = get_client(config)
            print(f"  Reflection client initialized")
        except Exception as e:
            print(f"Error: Cannot initialize reflection client: {e}")
            return 1

        print(f"Initializing reflection database: {args.db_path}")
        os.makedirs(args.db_path, exist_ok=True)
        reflection_db = ReflectionDatabase(
            db_path=args.db_path,
            embedding_model="all-MiniLM-L6-v2"
        )

    # Process log files
    log_files = sorted(logs_dir.glob("*.log"))
    print(f"\nFound {len(log_files)} log files")

    success_count = 0
    skip_count = 0
    error_count = 0
    filtered_count = 0

    for log_file in log_files:
        try:
            # Extract file index from log filename
            file_index = get_file_index(log_file.stem)
            if file_index is None:
                print(f"Warning: Cannot extract index from {log_file.name}, skipping")
                skip_count += 1
                continue

            # Apply split filtering
            if args.split_mode == "train" and file_index not in TRAIN_INDICES:
                filtered_count += 1
                continue
            elif args.split_mode == "test" and file_index not in TEST_INDICES:
                filtered_count += 1
                continue

            # Parse log file
            log_data = parse_log_file(log_file, has_reasoning=has_reasoning)
            family = log_data["family"]
            file_name = log_data["file_name"]

            if not family or not file_name:
                print(f"Warning: Cannot extract family or file_name from {log_file.name}, skipping")
                skip_count += 1
                continue

            # Get score
            if family not in scores_by_family:
                print(f"Warning: Family '{family}' not in score file, skipping {log_file.name}")
                skip_count += 1
                continue

            scores = scores_by_family[family]
            if file_index > len(scores):
                print(f"Warning: Index {file_index} exceeds scores for {family}, skipping")
                skip_count += 1
                continue

            score = scores[file_index - 1]

            # Check score threshold
            if score < args.score_threshold:
                skip_count += 1
                continue

            if args.dry_run:
                print(f"  {family}/{file_name} (index={file_index}, score={score})")
                success_count += 1
                continue

            # Read solution file
            solution_file = solutions_dir / log_file.name.replace(".log", "")
            code = ""

            if solution_file.exists():
                code = solution_file.read_text(encoding='utf-8', errors='ignore')
            else:
                solution_file = solutions_dir / f"{family}_{file_name}"
                if solution_file.exists():
                    code = solution_file.read_text(encoding='utf-8', errors='ignore')
                else:
                    code = log_data["extracted_code"]
                    if not code.strip():
                        print(f"Warning: No code found for {log_file.name}, skipping")
                        skip_count += 1
                        continue

            if not code.strip():
                skip_count += 1
                continue

            # Prepare data
            user_request = log_data["user_request"]
            reasoning = log_data["reasoning"] if log_data["reasoning"] else "N/A"

            if not user_request.strip():
                skip_count += 1
                continue

            print(f"\nProcessing {family}/{file_name} (index={file_index}, score={score})...")

            # Generate reflection
            try:
                reflection_raw = generate_reflection(
                    reflection_client=reflection_client,
                    user_request=user_request,
                    reasoning=reasoning,
                    code=code,
                    score=score
                )

                reflection_parsed = parse_reflection_response(reflection_raw)

                reflection_id = reflection_db.store_reflection(
                    family=family,
                    file_name=file_name,
                    user_request=user_request,
                    reasoning=reasoning,
                    code=code,
                    score=score,
                    reflection_raw=reflection_raw,
                    reflection_parsed=reflection_parsed,
                    source=args.source_model_name
                )

                print(f"  Reflection stored (ID: {reflection_id[:8]}...)")
                success_count += 1

            except Exception as e:
                print(f"  Error generating/storing reflection: {e}")
                error_count += 1

        except Exception as e:
            print(f"Error processing {log_file.name}: {e}")
            error_count += 1

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Split mode: {args.split_mode}")
    print(f"Source model: {args.source_model_name}")
    print(f"Database path: {args.db_path}")
    print("-" * 60)
    print(f"Successful: {success_count}")
    print(f"Filtered (wrong split): {filtered_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Total log files: {len(log_files)}")
    print("=" * 60)

    if not args.dry_run:
        stats = reflection_db.get_collection_stats()
        print(f"\nDatabase statistics:")
        print(f"  Total reflections: {stats['total_reflections']}")
        print(f"  Database path: {stats['db_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
