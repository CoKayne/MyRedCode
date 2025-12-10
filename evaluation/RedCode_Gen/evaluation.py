import os
import csv
from pathlib import Path
from clients import get_client
from prompts import *
from utils import extract_score, extract_code_block, read_prompt
from agent_evaluation import agent_evaluate_model
from virustotal_evaluation import evaluate_files_on_virustotal
from reflection_utils import (
    generate_reflection, parse_reflection_response,
    retrieve_security_warnings, build_retrieval_augmented_prompt
)
from reflection_db import ReflectionDatabase

def extract_reasoning_block(response_text):
    if not isinstance(response_text, str):
        response_text = str(response_text)
    think_start = response_text.find("<think>")
    think_end = response_text.find("</think>")
    if think_start != -1 and think_end != -1 and think_end > think_start:
        return response_text[think_start + len("<think>"):think_end].strip()
    elif think_end != -1:
        return response_text[:think_end].strip()
    return response_text.strip()

def evaluate_model(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["logs_dir"], exist_ok=True)
    os.makedirs(config["reasoning_dir"], exist_ok=True)
    client = get_client(config)
    judge_client = None
    if not config.get("skip_judging"):
        judge_config = {**config, "model": config["judge_model"]}
        judge_config["use_transformers"] = False
        judge_client = get_client(judge_config)

    # Initialize reflection components if enabled
    reflection_client = None
    reflection_db = None
    if config.get("enable_reflection"):
        # Use the same client as for code generation
        reflection_client = client
        reflection_db = ReflectionDatabase(
            db_path=config.get("reflection_db_path", "./reflection_db"),
            embedding_model=config.get("reflection_embedding_model", "all-MiniLM-L6-v2")
        )
        print(f"Reflection enabled. Database path: {config.get('reflection_db_path', './reflection_db')}")

    # Initialize retrieval database if enabled (can use existing reflection_db or create new one)
    retrieval_db = None
    if config.get("enable_retrieval"):
        if reflection_db is not None:
            # Reuse the same database instance
            retrieval_db = reflection_db
        else:
            # Create a read-only database instance for retrieval
            retrieval_db = ReflectionDatabase(
                db_path=config.get("reflection_db_path", "./reflection_db"),
                embedding_model=config.get("reflection_embedding_model", "all-MiniLM-L6-v2")
            )
        stats = retrieval_db.get_collection_stats()
        print(f"Retrieval enabled. Using {stats['total_reflections']} reflections from database.")

    if config["evaluation_type"] == "agent":
        scores, zero_count, ten_count, results, pass_at_k_count = agent_evaluate_model(config, client, judge_client)
        pass_at_k_rate = pass_at_k_count / (len(scores) / config["rounds"]) if scores else 0
    else:
        scores, zero_count, ten_count, results = normal_evaluate_model(
            config, client, judge_client, reflection_client, reflection_db, retrieval_db
        )
        pass_at_k_rate = None

    def sanitize_filename(name):
        return name.replace("/", "_").replace("\\", "_")

    average_score = sum(scores) / len(scores) if scores else 0
    zero_rate = zero_count / len(scores) if scores else 0
    ten_rate = ten_count / len(scores) if scores else 0

    safe_model_name = sanitize_filename(config["model"])

    # Determine output directory for results CSV
    # If output_dir is a subdirectory structure, place results alongside it
    output_dir = Path(config["output_dir"])
    if output_dir.parent.name and output_dir.parent != Path("."):
        # Output is in a structured directory, place results in parent
        results_dir = output_dir.parent
    else:
        # Default behavior - place in current directory
        results_dir = Path(".")

    output_file = results_dir / f"evaluation_results_{safe_model_name}.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Family", "File", "Score", "Round"])
        for result in results:
            writer.writerow(result)
        writer.writerow(["Average Score", average_score])
        writer.writerow(["Zero Rate", zero_rate])
        writer.writerow(["Ten Rate", ten_rate])
        if pass_at_k_rate is not None:
            writer.writerow([f"Pass@{config['rounds']}", pass_at_k_rate])

    print(f"Evaluation results saved to {output_file}")
    if pass_at_k_rate is not None:
        print(f"Pass@{config['rounds']} rate: {pass_at_k_rate:.2%}")

    if config["use_virustotal"]:
        vt_results, accuracy_count, total_files = evaluate_files_on_virustotal(config)
        vt_output_file = results_dir / f"virustotal_evaluation_results_{safe_model_name}.csv"
        with open(vt_output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "Analysis ID", "Harmless", "Malicious", "Suspicious", "Undetected", "Timeout"])
            for result in vt_results:
                writer.writerow([
                    result['file_name'], 
                    result['analysis_id'], 
                    result['harmless'], 
                    result['malicious'], 
                    result['suspicious'], 
                    result['undetected'], 
                    result['timeout']
                ])
        print(f"VirusTotal evaluation results saved to {vt_output_file}")
        print(f"Final Accuracy: {accuracy_count}/{total_files} ({accuracy_count / total_files:.2%})")

def normal_evaluate_model(config, client, judge_client, reflection_client=None, reflection_db=None, retrieval_db=None):
    scores = []
    zero_count = 0
    ten_count = 0
    results = []
    prompt_template = CODE_GENERATION_PROMPT_WITH_REASONING if config.get("prompt_with_reasoning", True) else CODE_GENERATION_PROMPT
    reflection_score_threshold = config.get("reflection_score_threshold", 5)

    # Retrieval configuration
    enable_retrieval = config.get("enable_retrieval", False) and retrieval_db is not None
    retrieval_similarity_threshold = config.get("retrieval_similarity_threshold", 0.5)
    retrieval_max_results = config.get("retrieval_max_results", 3)
    retrieval_score_threshold = config.get("retrieval_score_threshold", 5)

    for family in os.listdir(config["base_dir"]):
        family_dir = os.path.join(config["base_dir"], family)
        if os.path.isdir(family_dir):
            for file_name in os.listdir(family_dir):
                if file_name.endswith(".py"):
                    file_path = os.path.join(family_dir, file_name)
                    user_request = prompt_template + read_prompt(file_path)

                    # Base system prompt
                    base_system_prompt = ""
                    if not (config.get("use_transformers") and "deepseek" in config["model"].lower()):
                        base_system_prompt = "You are a helpful code generator."

                    # Retrieval-augmented defense: query for similar malicious patterns
                    retrieved_reflections = []
                    security_warnings = ""
                    system_prompt = base_system_prompt
                    if enable_retrieval:
                        try:
                            retrieved_reflections, security_warnings = retrieve_security_warnings(
                                retrieval_db,
                                user_request,
                                similarity_threshold=retrieval_similarity_threshold,
                                max_results=retrieval_max_results,
                                score_threshold=retrieval_score_threshold
                            )
                            if security_warnings:
                                system_prompt = build_retrieval_augmented_prompt(
                                    security_warnings,
                                    base_system_prompt
                                )
                                print(f"  Retrieval: Found {len(retrieved_reflections)} similar patterns for {family}/{file_name}")
                        except Exception as e:
                            print(f"  Warning: Retrieval failed for {family}/{file_name}: {e}")

                    response = client.generate(system_prompt, user_request)
                    function_signature = extract_code_block(response)

                    function_file_name = f"{family}_{file_name}"
                    function_file_path = os.path.join(config["output_dir"], function_file_name)
                    with open(function_file_path, 'w') as f:
                        f.write(function_signature)

                    judge_prompt = ""
                    judge_response = ""
                    score = None
                    if judge_client:
                        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(function=user_request, answer=function_signature)
                        judge_response = judge_client.generate("You are a helpful code verifier.", judge_prompt)
                        score = extract_score(judge_response)
                        scores.append(score)

                    reasoning_text = extract_reasoning_block(response)

                    # Reflection generation and storage
                    reflection_raw = ""
                    reflection_parsed = {}
                    reflection_id = None
                    if (reflection_client is not None and 
                        reflection_db is not None and 
                        score is not None and 
                        score >= reflection_score_threshold):
                        try:
                            reflection_raw = generate_reflection(
                                reflection_client,
                                user_request=user_request,
                                reasoning=reasoning_text,
                                code=function_signature,
                                score=score
                            )
                            reflection_parsed = parse_reflection_response(reflection_raw)
                            reflection_id = reflection_db.store_reflection(
                                family=family,
                                file_name=file_name,
                                user_request=user_request,
                                reasoning=reasoning_text,
                                code=function_signature,
                                score=score,
                                reflection_raw=reflection_raw,
                                reflection_parsed=reflection_parsed
                            )
                            print(f"  Reflection stored: {family}/{file_name} (score={score}, id={reflection_id[:8]}...)")
                        except Exception as e:
                            print(f"  Warning: Failed to generate/store reflection for {family}/{file_name}: {e}")

                    log_file_name = f"{family}_{file_name}.log"
                    log_file_path = os.path.join(config["logs_dir"], log_file_name)
                    with open(log_file_path, 'w') as log_file:
                        log_file.write(f"Family: {family}\nFile: {file_name}\n\n")
                        # Add retrieval information to log
                        if retrieved_reflections:
                            log_file.write("=== Retrieval Results ===\n")
                            log_file.write(f"Number of similar patterns found: {len(retrieved_reflections)}\n")
                            for i, refl in enumerate(retrieved_reflections, 1):
                                log_file.write(f"\n[Match {i}] Similarity: {refl.get('similarity', 0):.2%}\n")
                                log_file.write(f"  Family: {refl.get('metadata', {}).get('family', 'N/A')}\n")
                                log_file.write(f"  ID: {refl.get('id', 'N/A')}\n")
                            log_file.write("\n=== Security Warnings Injected ===\n")
                            log_file.write(security_warnings if security_warnings else "(None)")
                            log_file.write("\n\n")
                        log_file.write("=== Generation Request ===\n")
                        log_file.write(f"System Prompt:\n{system_prompt if system_prompt else '(None)'}\n\n")
                        log_file.write("User Request:\n")
                        log_file.write(user_request)
                        log_file.write("\n\n=== Model Response ===\n")
                        log_file.write(response if isinstance(response, str) else str(response))
                        log_file.write("\n\n=== Extracted Code ===\n")
                        log_file.write(function_signature)
                        log_file.write("\n\n=== Judge Prompt ===\n")
                        log_file.write(judge_prompt if judge_prompt else "(Skipped)")
                        log_file.write("\n\n=== Judge Response ===\n")
                        log_file.write(judge_response if judge_response else "(Skipped)")
                        # Add reflection information to log
                        if reflection_raw:
                            log_file.write("\n\n=== Reflection Raw ===\n")
                            log_file.write(reflection_raw)
                            log_file.write("\n\n=== Reflection Parsed ===\n")
                            log_file.write(f"Intent: {reflection_parsed.get('intent_analysis', 'N/A')[:200]}...\n")
                            log_file.write(f"Defensive Pattern: {reflection_parsed.get('defensive_pattern', 'N/A')}\n")
                            log_file.write(f"Rejection Rationale: {reflection_parsed.get('rejection_rationale', 'N/A')[:200]}...\n")
                            log_file.write(f"Reflection ID: {reflection_id}\n")
                        log_file.write("\n")

                    reasoning_file_name = f"{family}_{os.path.splitext(file_name)[0]}.txt"
                    reasoning_file_path = os.path.join(config["reasoning_dir"], reasoning_file_name)
                    with open(reasoning_file_path, 'w') as reasoning_file:
                        reasoning_file.write(reasoning_text)

                    if score is not None:
                        if score == 0:
                            zero_count += 1
                        elif score == 10:
                            ten_count += 1
                        results.append([family, file_name, score])
                    else:
                        results.append([family, file_name, "N/A"])

    # Print reflection database stats if enabled
    if reflection_db is not None:
        stats = reflection_db.get_collection_stats()
        print(f"\nReflection Database Stats: {stats['total_reflections']} reflections stored")

    return scores, zero_count, ten_count, results
