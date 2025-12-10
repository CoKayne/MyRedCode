import os
import argparse
from dotenv import load_dotenv

load_dotenv()

def str_to_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

def get_config():
    parser = argparse.ArgumentParser(description="RedCode-Gen")
    parser.add_argument("--base_dir", default=os.getenv("BASE_DIR", "./dataset/RedCode-Gen"), help="Directory containing the prompts")
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "./solutions"), help="Directory to store solutions")
    parser.add_argument("--model", default=os.getenv("MODEL", "gpt-3.5-turbo"), help="Model to evaluate")
    parser.add_argument("--judge_model", default=os.getenv("JUDGE_MODEL", "gpt-4"), help="Model to use for judging")
    parser.add_argument("--evaluation_type", choices=["normal", "agent"], default="normal", help="Type of evaluation to perform")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds for agent evaluation")
    parser.add_argument("--use_virustotal", action="store_true", help="Use VirusTotal for additional evaluation")
    parser.add_argument("--skip_judging", action="store_true", help="Skip judge model evaluation and only generate code")
    parser.add_argument("--use_transformers", action="store_true", help="Use local Hugging Face Transformers for generation")
    parser.add_argument("--transformers_device_map", default=os.getenv("TRANSFORMERS_DEVICE_MAP", "cuda"), help="Device map for Transformers models (auto, sequential, cuda:0, etc.)")
    parser.add_argument("--transformers_dtype", default=os.getenv("TRANSFORMERS_DTYPE", "bfloat16"), help="Torch dtype for Transformers models (float16, bfloat16, float32, auto)")
    parser.add_argument("--transformers_max_new_tokens", type=int, default=int(os.getenv("TRANSFORMERS_MAX_NEW_TOKENS", "2048")), help="Max new tokens to generate with Transformers models")
    parser.add_argument("--transformers_trust_remote_code", default=os.getenv("TRANSFORMERS_TRUST_REMOTE_CODE", "true"), help="Allow Transformers to trust remote code when loading models")
    parser.add_argument("--transformers_load_in_8bit", default=os.getenv("TRANSFORMERS_LOAD_IN_8BIT", "false"), help="Load Transformers model in 8-bit (requires bitsandbytes)")
    parser.add_argument("--logs_dir", default=os.getenv("LOGS_DIR", "./logs"), help="Directory to store raw model conversations")
    parser.add_argument("--reasoning_dir", default=os.getenv("REASONING_DIR", "./reasoning"), help="Directory to store extracted reasoning blocks")
    parser.add_argument("--prompt_with_reasoning", default=os.getenv("PROMPT_WITH_REASONING", "true"), help="Use prompt variant that asks for reasoning before code")
    # Thinking mode (Qwen3 models)
    parser.add_argument("--enable_thinking", action="store_true", help="Enable model thinking mode (Qwen3 only)")
    # Reflection-related arguments
    parser.add_argument("--enable_reflection", action="store_true", help="Enable reflection generation and storage")
    parser.add_argument("--reflection_db_path", default=os.getenv("REFLECTION_DB_PATH", "./reflection_db"), help="Path to the ChromaDB reflection database")
    parser.add_argument("--reflection_embedding_model", default=os.getenv("REFLECTION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"), help="Sentence-transformers model for embeddings")
    parser.add_argument("--reflection_score_threshold", type=int, default=int(os.getenv("REFLECTION_SCORE_THRESHOLD", "5")), help="Minimum judge score to generate reflection (default: 5)")
    # Retrieval-augmented defense arguments
    parser.add_argument("--enable_retrieval", action="store_true", help="Enable retrieval-augmented defense using reflection database")
    parser.add_argument("--retrieval_similarity_threshold", type=float, default=float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.5")), help="Minimum similarity score to include retrieved reflections (default: 0.5)")
    parser.add_argument("--retrieval_max_results", type=int, default=int(os.getenv("RETRIEVAL_MAX_RESULTS", "3")), help="Maximum number of similar reflections to retrieve (default: 3)")
    parser.add_argument("--retrieval_score_threshold", type=int, default=int(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "5")), help="Minimum judge score of stored reflections to use for retrieval (default: 5)")
    args = parser.parse_args()

    transformers_trust_remote_code = str_to_bool(args.transformers_trust_remote_code)
    transformers_load_in_8bit = str_to_bool(args.transformers_load_in_8bit)
    prompt_with_reasoning = str_to_bool(args.prompt_with_reasoning)

    config = {
        "base_dir": args.base_dir,
        "output_dir": args.output_dir,
        "model": args.model,
        "judge_model": args.judge_model,
        "evaluation_type": args.evaluation_type,
        "rounds": args.rounds,
        "use_virustotal": args.use_virustotal,
        "skip_judging": args.skip_judging,
        "use_transformers": args.use_transformers,
        "transformers_device_map": args.transformers_device_map,
        "transformers_dtype": args.transformers_dtype,
        "transformers_max_new_tokens": args.transformers_max_new_tokens,
        "transformers_trust_remote_code": transformers_trust_remote_code,
        "transformers_load_in_8bit": transformers_load_in_8bit,
        "logs_dir": args.logs_dir,
        "reasoning_dir": args.reasoning_dir,
        "prompt_with_reasoning": prompt_with_reasoning,
        "enable_thinking": args.enable_thinking,
        # Reflection configuration
        "enable_reflection": args.enable_reflection,
        "reflection_db_path": args.reflection_db_path,
        "reflection_embedding_model": args.reflection_embedding_model,
        "reflection_score_threshold": args.reflection_score_threshold,
        # Retrieval configuration
        "enable_retrieval": args.enable_retrieval,
        "retrieval_similarity_threshold": args.retrieval_similarity_threshold,
        "retrieval_max_results": args.retrieval_max_results,
        "retrieval_score_threshold": args.retrieval_score_threshold,
        # API keys
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "virustotal_api_key": os.getenv("VIRUSTOTAL_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    }

    return config
