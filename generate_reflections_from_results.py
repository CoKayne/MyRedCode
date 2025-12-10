#!/usr/bin/env python3
"""
從結果資料夾中的資料生成 reflection 並儲存到資料庫。

此腳本會：
1. 解析 log 檔案，提取 user prompt, function prototype, written code
2. 讀取 parsed code（從 solutions 資料夾）
3. 從 score 檔案讀取每個測試案例的分數
4. 使用 reflection client 生成 reflection
5. 將 reflection 儲存到資料庫

支援：
- 自動檢測結果資料夾結構
- 支援有 reasoning 和無 reasoning 的結果
- 可控制是否期望有 reasoning
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加 evaluation/RedCode_Gen 到路徑
sys.path.insert(0, str(Path(__file__).parent / "evaluation" / "RedCode_Gen"))

from clients import get_client
from reflection_utils import generate_reflection, parse_reflection_response, extract_function_prototype
from reflection_db import ReflectionDatabase


def detect_results_structure(results_dir: Path) -> Dict[str, Path]:
    """
    自動檢測結果資料夾的結構。
    
    Returns:
        dict with keys: logs_dir, solutions_dir, score_file, reasoning_dir (optional)
    """
    structure = {
        "logs_dir": None,
        "solutions_dir": None,
        "score_file": None,
        "reasoning_dir": None
    }
    
    # 尋找 logs 資料夾（可能的命名模式）
    for pattern in ["logs_*", "logs", "*logs*"]:
        for path in results_dir.glob(pattern):
            if path.is_dir():
                structure["logs_dir"] = path
                break
        if structure["logs_dir"]:
            break
    
    # 尋找 solutions 資料夾
    for pattern in ["solutions_*", "solutions", "*solutions*"]:
        for path in results_dir.glob(pattern):
            if path.is_dir():
                structure["solutions_dir"] = path
                break
        if structure["solutions_dir"]:
            break
    
    # 尋找 score 檔案
    score_file = results_dir / "score"
    if score_file.exists():
        structure["score_file"] = score_file
    
    # 尋找 reasoning 資料夾（可選）
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
    解析 log 檔案，提取相關資訊。
    
    Args:
        log_path: Log 檔案路徑
        has_reasoning: 是否期望有 reasoning 區塊
    
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
    
    # 提取 Family
    family_match = re.search(r'^Family:\s*(.+)$', log_text, re.MULTILINE)
    if family_match:
        result["family"] = family_match.group(1).strip()
    
    # 提取 File
    file_match = re.search(r'^File:\s*(.+)$', log_text, re.MULTILINE)
    if file_match:
        result["file_name"] = file_match.group(1).strip()
    
    # 提取 User Request（從 "=== Generation Request ===" 到 "=== Model Response ==="）
    gen_request_start = log_text.find("=== Generation Request ===")
    model_response_start = log_text.find("=== Model Response ===")
    
    if gen_request_start != -1 and model_response_start != -1:
        user_request_section = log_text[gen_request_start:model_response_start]
        # 提取 User Request 部分（在 "User Request:" 之後，包含所有內容直到 Model Response）
        user_request_match = re.search(r'User Request:\s*(.+)$', user_request_section, re.DOTALL)
        if user_request_match:
            result["user_request"] = user_request_match.group(1).strip()
    
    # 提取 Model Response
    if model_response_start != -1:
        model_response_section = log_text[model_response_start:]
        # 找到 "=== Extracted Code ===" 或 "=== Judge Prompt ===" 的位置
        extracted_code_start = model_response_section.find("=== Extracted Code ===")
        judge_prompt_start = model_response_section.find("=== Judge Prompt ===")
        
        end_pos = extracted_code_start if extracted_code_start != -1 else judge_prompt_start
        if end_pos != -1:
            result["model_response"] = model_response_section[:end_pos].strip()
        else:
            result["model_response"] = model_response_section.strip()
    
    # 提取 Extracted Code
    extracted_code_start = log_text.find("=== Extracted Code ===")
    if extracted_code_start != -1:
        extracted_code_section = log_text[extracted_code_start:]
        judge_prompt_start = extracted_code_section.find("=== Judge Prompt ===")
        if judge_prompt_start != -1:
            extracted_code_text = extracted_code_section[len("=== Extracted Code ==="):judge_prompt_start].strip()
            result["extracted_code"] = extracted_code_text
    
    # 提取 Reasoning
    if has_reasoning:
        # 優先從 <think> 標籤提取
        reasoning_match = re.search(r'<think>(.*?)</think>', log_text, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            # 如果沒有閉合標籤，嘗試從 Model Response 中提取
            if result["model_response"]:
                open_tag_pos = result["model_response"].find("<think>")
                if open_tag_pos != -1:
                    # 找到下一個 ``` 或 </think> 的位置
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
        # 如果不需要 reasoning，設為 "N/A"
        result["reasoning"] = "N/A"
    
    return result


def parse_score_file(score_path: Path) -> Dict[str, List[int]]:
    """
    解析 score 檔案，提取每個 family 的分數列表。
    
    Returns:
        dict mapping family name to list of scores
    """
    score_text = score_path.read_text(encoding='utf-8')
    scores_by_family = {}
    
    # 解析格式：family: [score1, score2, ...]
    pattern = r'^(\w+):\s*\[(.*?)\]$'
    for line in score_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('---'):
            continue
        
        match = re.match(pattern, line)
        if match:
            family = match.group(1)
            scores_str = match.group(2)
            # 解析分數列表
            scores = [int(s.strip()) for s in scores_str.split(',') if s.strip()]
            scores_by_family[family] = scores
    
    return scores_by_family


def get_file_index(file_name: str) -> Optional[int]:
    """
    從檔案名稱提取索引（例如 ddos_1_udp_flood.py -> 1）。
    
    Returns:
        索引（1-based），如果無法提取則返回 None
    """
    # 嘗試匹配 pattern: family_family_N_name.py
    match = re.search(r'_(\d+)_', file_name)
    if match:
        return int(match.group(1))
    
    # 嘗試匹配 pattern: family_name_N.py
    match = re.search(r'_(\d+)\.py$', file_name)
    if match:
        return int(match.group(1))
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="從結果資料夾生成 reflection 並儲存到資料庫"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="結果資料夾路徑（如果未指定，會自動檢測 RESULTS 資料夾下的所有結果）"
    )
    parser.add_argument(
        "--results_name",
        type=str,
        default=None,
        help="結果資料夾名稱（例如：llama_no_reasoning_results, llama_reasoning_results）。如果指定，會從 RESULTS/{results_name} 讀取"
    )
    parser.add_argument(
        "--has_reasoning",
        action="store_true",
        help="結果包含 reasoning（會從 <think> 標籤提取）。如果未指定，會自動檢測"
    )
    parser.add_argument(
        "--no_reasoning",
        action="store_true",
        help="明確指定結果不包含 reasoning（會設為 N/A）"
    )
    parser.add_argument(
        "--reflection_model",
        type=str,
        default="gpt-4o",
        help="用於生成 reflection 的模型（例如：gpt-4o, claude-3-5-sonnet, Qwen/Qwen2.5-7B-Instruct）"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="./reflection_db",
        help="Reflection 資料庫路徑"
    )
    parser.add_argument(
        "--score_threshold",
        type=int,
        default=0,
        help="只為分數 >= threshold 的測試案例生成 reflection（預設：0，即全部生成）"
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key（如果使用 OpenAI 模型）"
    )
    parser.add_argument(
        "--anthropic_api_key",
        type=str,
        default=None,
        help="Anthropic API key（如果使用 Claude 模型）"
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key（如果使用 Gemini 模型）"
    )
    parser.add_argument(
        "--use_transformers",
        action="store_true",
        help="使用 transformers（本地模型）而非 API"
    )
    parser.add_argument(
        "--transformers_device_map",
        type=str,
        default="auto",
        help="Transformers 模型的 device_map（預設：auto）"
    )
    parser.add_argument(
        "--transformers_dtype",
        type=str,
        default="float16",
        help="Transformers 模型的資料類型（預設：float16）"
    )
    parser.add_argument(
        "--transformers_max_new_tokens",
        type=int,
        default=2048,
        help="Transformers 模型的最大新 token 數（預設：2048）"
    )
    parser.add_argument(
        "--transformers_load_in_8bit",
        action="store_true",
        help="使用 8-bit 量化載入 transformers 模型"
    )
    parser.add_argument(
        "--transformers_trust_remote_code",
        action="store_true",
        help="信任遠端程式碼（用於某些 transformers 模型）"
    )
    parser.add_argument(
        "--prompt_with_reasoning",
        action="store_true",
        help="啟用 reasoning prompt（用於某些模型）"
    )
    
    args = parser.parse_args()
    
    # 設定 API keys
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key
    if args.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_api_key
    
    # 確定結果資料夾
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.results_name:
        results_dir = Path("RESULTS") / args.results_name
    else:
        # 預設使用 llama_no_reasoning_results
        results_dir = Path("RESULTS/llama_no_reasoning_results")
    
    if not results_dir.exists():
        print(f"錯誤：結果資料夾不存在：{results_dir}")
        print("\n可用的結果資料夾：")
        results_base = Path("RESULTS")
        if results_base.exists():
            for d in sorted(results_base.iterdir()):
                if d.is_dir():
                    print(f"  - {d.name}")
        return 1
    
    # 自動檢測資料夾結構
    print(f"正在檢測結果資料夾結構：{results_dir}")
    structure = detect_results_structure(results_dir)
    
    logs_dir = structure["logs_dir"]
    solutions_dir = structure["solutions_dir"]
    score_file = structure["score_file"]
    reasoning_dir = structure["reasoning_dir"]
    
    if not logs_dir:
        print(f"錯誤：找不到 logs 資料夾在 {results_dir}")
        return 1
    if not solutions_dir:
        print(f"錯誤：找不到 solutions 資料夾在 {results_dir}")
        return 1
    if not score_file:
        print(f"錯誤：找不到 score 檔案在 {results_dir}")
        return 1
    
    print(f"  ✓ Logs 資料夾：{logs_dir}")
    print(f"  ✓ Solutions 資料夾：{solutions_dir}")
    print(f"  ✓ Score 檔案：{score_file}")
    if reasoning_dir:
        print(f"  ✓ Reasoning 資料夾：{reasoning_dir}")
    
    # 自動檢測是否有 reasoning
    has_reasoning = None
    if args.has_reasoning:
        has_reasoning = True
    elif args.no_reasoning:
        has_reasoning = False
    else:
        # 自動檢測：檢查資料夾名稱或檢查一個 log 檔案
        if "no_reasoning" in results_dir.name.lower():
            has_reasoning = False
        elif "reasoning" in results_dir.name.lower() and "no_reasoning" not in results_dir.name.lower():
            has_reasoning = True
        else:
            # 檢查第一個 log 檔案
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                sample_log = log_files[0].read_text(encoding='utf-8', errors='ignore')
                has_reasoning = "<think>" in sample_log
                print(f"  自動檢測：{'有' if has_reasoning else '無'} reasoning")
            else:
                has_reasoning = False
    
    print(f"  使用模式：{'有 reasoning' if has_reasoning else '無 reasoning'}")
    
    # 解析 score 檔案
    print("\n正在解析 score 檔案...")
    scores_by_family = parse_score_file(score_file)
    print(f"已解析 {len(scores_by_family)} 個 families 的分數")
    
    # 初始化 reflection client 和 database
    print(f"\n正在初始化 reflection client（模型：{args.reflection_model}）...")
    
    # 自動檢測是否應該使用 transformers
    use_transformers = args.use_transformers
    if not use_transformers:
        model_lower = args.reflection_model.lower()
        
        # 檢查是否為 Ollama 模型
        is_ollama = (":" in args.reflection_model or 
                     args.reflection_model.startswith(("llama", "mistral", "gemma", "qwen", "phi")))
        
        # 檢查是否為已知的 API 模型
        is_api_model = (model_lower.startswith("gpt") or 
                       model_lower.startswith("claude") or 
                       model_lower.startswith("gemini") or
                       model_lower.startswith("models/"))
        
        if not is_ollama and not is_api_model:
            print(f"提示：模型 '{args.reflection_model}' 不是已知的 API 或 Ollama 模型。")
            print("如果這是本地 transformers 模型，請使用 --use_transformers 參數。")
    
    config = {
        "model": args.reflection_model,
        "use_transformers": use_transformers,
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "transformers_device_map": args.transformers_device_map,
        "transformers_dtype": args.transformers_dtype,
        "transformers_max_new_tokens": args.transformers_max_new_tokens,
        "transformers_load_in_8bit": args.transformers_load_in_8bit,
        "transformers_trust_remote_code": args.transformers_trust_remote_code,
        "prompt_with_reasoning": args.prompt_with_reasoning,
    }
    
    try:
        reflection_client = get_client(config)
        print(f"✓ Reflection client 初始化成功（類型：{'Transformers' if use_transformers else 'API'}）")
    except Exception as e:
        print(f"錯誤：無法初始化 reflection client：{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"正在初始化 reflection database（路徑：{args.db_path}）...")
    reflection_db = ReflectionDatabase(
        db_path=args.db_path,
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # 處理所有 log 檔案
    log_files = sorted(logs_dir.glob("*.log"))
    print(f"\n找到 {len(log_files)} 個 log 檔案")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for log_file in log_files:
        try:
            # 解析 log 檔案
            log_data = parse_log_file(log_file, has_reasoning=has_reasoning)
            family = log_data["family"]
            file_name = log_data["file_name"]
            
            if not family or not file_name:
                print(f"警告：無法從 {log_file.name} 提取 family 或 file_name，跳過")
                skip_count += 1
                continue
            
            # 獲取檔案索引
            file_index = get_file_index(log_file.stem)
            if file_index is None:
                print(f"警告：無法從 {log_file.name} 提取檔案索引，跳過")
                skip_count += 1
                continue
            
            # 獲取分數（索引是 1-based，但列表是 0-based）
            if family not in scores_by_family:
                print(f"警告：family '{family}' 不在 score 檔案中，跳過 {log_file.name}")
                skip_count += 1
                continue
            
            scores = scores_by_family[family]
            if file_index > len(scores):
                print(f"警告：檔案索引 {file_index} 超出 {family} 的分數列表長度，跳過 {log_file.name}")
                skip_count += 1
                continue
            
            score = scores[file_index - 1]  # 轉換為 0-based 索引
            
            # 檢查分數閾值
            if score < args.score_threshold:
                print(f"跳過 {family}/{file_name}：分數 {score} < 閾值 {args.score_threshold}")
                skip_count += 1
                continue
            
            # 讀取 solution 檔案（parsed code）
            solution_file = solutions_dir / log_file.name.replace(".log", "")
            code = ""
            
            if solution_file.exists():
                code = solution_file.read_text(encoding='utf-8', errors='ignore')
            else:
                # 嘗試其他可能的檔案名稱格式
                solution_file = solutions_dir / f"{family}_{file_name}"
                if solution_file.exists():
                    code = solution_file.read_text(encoding='utf-8', errors='ignore')
                else:
                    # 如果找不到 solution 檔案，使用 extracted_code
                    code = log_data["extracted_code"]
                    if not code.strip():
                        print(f"警告：找不到 solution 檔案且 extracted_code 為空，跳過 {log_file.name}")
                        skip_count += 1
                        continue
            
            if not code.strip():
                print(f"警告：{log_file.name} 的 code 為空，跳過")
                skip_count += 1
                continue
            
            # 準備資料
            user_request = log_data["user_request"]
            reasoning = log_data["reasoning"] if log_data["reasoning"] else "N/A"
            
            # 如果 user_request 為空，嘗試從 log 中提取
            if not user_request.strip():
                # 重新讀取 log 檔案並嘗試提取
                log_text = log_file.read_text(encoding='utf-8', errors='ignore')
                gen_request_start = log_text.find("=== Generation Request ===")
                model_response_start = log_text.find("=== Model Response ===")
                
                if gen_request_start != -1 and model_response_start != -1:
                    user_request_section = log_text[gen_request_start:model_response_start]
                    user_request_match = re.search(r'User Request:\s*(.+)$', user_request_section, re.DOTALL)
                    if user_request_match:
                        user_request = user_request_match.group(1).strip()
                
                if not user_request.strip():
                    print(f"警告：無法提取 user_request from {log_file.name}，跳過")
                    skip_count += 1
                    continue
            
            print(f"\n處理 {family}/{file_name} (分數: {score}, reasoning: {'有' if reasoning != 'N/A' else '無'})...")
            
            # 生成 reflection
            try:
                reflection_raw = generate_reflection(
                    reflection_client=reflection_client,
                    user_request=user_request,
                    reasoning=reasoning,
                    code=code,
                    score=score
                )
                
                # 解析 reflection
                reflection_parsed = parse_reflection_response(reflection_raw)
                
                # 儲存到資料庫
                # 使用結果資料夾名稱作為來源標識（例如：llama_reasoning_results -> llama_reasoning）
                source_name = results_dir.name.replace("_results", "").replace("_result", "")
                
                reflection_id = reflection_db.store_reflection(
                    family=family,
                    file_name=file_name,
                    user_request=user_request,
                    reasoning=reasoning,
                    code=code,
                    score=score,
                    reflection_raw=reflection_raw,
                    reflection_parsed=reflection_parsed,
                    source=source_name
                )
                
                print(f"  ✓ Reflection 已儲存（ID: {reflection_id[:8]}...）")
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ 生成或儲存 reflection 時發生錯誤：{e}")
                error_count += 1
                import traceback
                traceback.print_exc()
        
        except Exception as e:
            print(f"處理 {log_file.name} 時發生錯誤：{e}")
            error_count += 1
            import traceback
            traceback.print_exc()
    
    # 輸出統計資訊
    print("\n" + "="*60)
    print("處理完成！")
    print(f"成功：{success_count}")
    print(f"跳過：{skip_count}")
    print(f"錯誤：{error_count}")
    print(f"總計：{len(log_files)}")
    print("="*60)
    
    # 顯示資料庫統計
    stats = reflection_db.get_collection_stats()
    print(f"\n資料庫統計：")
    print(f"  總 reflection 數量：{stats['total_reflections']}")
    print(f"  資料庫路徑：{stats['db_path']}")
    print(f"  Embedding 模型：{stats['embedding_model']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
