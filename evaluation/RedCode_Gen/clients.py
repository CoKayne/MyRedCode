from abc import ABC, abstractmethod
from openai import OpenAI
import anthropic
import ollama
from google import genai
from google.genai import types
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import threading
import time

class LLMClient(ABC):
    @abstractmethod
    def generate(self, system, user_request):
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key, model):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system, user_request):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_request},
            ],
            temperature=0.8,
            max_tokens=2000,
            top_p=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content

class AnthropicClient(LLMClient):
    def __init__(self, api_key, model):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system, user_request):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.8,
            system=system,
            messages=[
                {"role": "user", "content": user_request}
            ]
        )
        return message.content[0].text

class TransformersClient(LLMClient):
    _cache = {}
    _lock = threading.Lock()

    def __init__(self, model, torch_dtype="float16", device_map="auto", max_new_tokens=1024, load_in_8bit=False, trust_remote_code=False, temperature=0.6, top_p=0.95):
        self.model_name = model
        self.torch_dtype = torch_dtype
        self.device_map = device_map or "auto"
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
        self.temperature = temperature
        self.top_p = top_p

        cache_key = (self.model_name, self.torch_dtype, self.device_map, self.load_in_8bit, self.trust_remote_code)
        with self._lock:
            if cache_key not in self._cache:
                dtype = self._resolve_dtype(self.torch_dtype)
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"

                model_kwargs = {
                    "trust_remote_code": self.trust_remote_code,
                }
                if dtype is not None:
                    model_kwargs["torch_dtype"] = dtype
                if self.device_map:
                    model_kwargs["device_map"] = self.device_map
                if self.load_in_8bit:
                    model_kwargs["load_in_8bit"] = True

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                text_gen = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer
                )
                self._cache[cache_key] = (text_gen, tokenizer)

        self.pipeline, self.tokenizer = self._cache[cache_key]

    def _resolve_dtype(self, dtype):
        if not dtype or dtype == "auto":
            return None
        value = str(dtype).strip().lower()
        if value in {"float16", "fp16", "half"}:
            return torch.float16
        if value in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if value in {"float32", "fp32"}:
            return torch.float32
        return None

    def generate(self, system, user_request):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_request})

        prompt = None
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                prompt = None

        if prompt is None:
            prompt = f"{system}\n\nUser:\n{user_request}\nAssistant:"

        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False
        )
        return outputs[0]["generated_text"].strip()

class OllamaClient(LLMClient):
    def __init__(self, model):
        self.model = model

    def generate(self, system, user_request):
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_request},
            ],
            options={
                "temperature": 0.8,
                "num_predict": 2000,
            }
        )
        return response['message']['content']

class GeminiClient(LLMClient):
    def __init__(self, api_key, model, max_retries=3, retry_delay=5):
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, system, user_request):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_request,
                    config=types.GenerateContentConfig(
                        temperature=1.0,
                        top_p=0.95,
                        max_output_tokens=2000,
                        system_instruction=system,
                    ),
                )
                break
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                message = str(e)
                retriable = status_code in (429, 500, 502, 503) or "UNAVAILABLE" in message.upper()
                last_error = e
                if attempt == self.max_retries - 1 or not retriable:
                    raise
                time.sleep(self.retry_delay)
        else:
            raise last_error if last_error else RuntimeError("Gemini request failed")

        text = getattr(response, "text", None)
        if not text:
            text = getattr(response, "output_text", None)
        if not text and hasattr(response, "candidates"):
            parts = []
            for candidate in response.candidates:
                content = getattr(candidate, "content", None)
                if not content or not hasattr(content, "parts"):
                    continue
                for part in content.parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        parts.append(part_text)
            text = "\n".join(parts) if parts else None
        if not text:
            text = str(response)
        return text

def get_client(config):
    model = config["model"]
    
    if config.get("use_transformers"):
        return TransformersClient(
            model=model,
            torch_dtype=config.get("transformers_dtype"),
            device_map=config.get("transformers_device_map"),
            max_new_tokens=config.get("transformers_max_new_tokens", 1024),
            load_in_8bit=config.get("transformers_load_in_8bit"),
            trust_remote_code=config.get("transformers_trust_remote_code"),
        )

    # Local Ollama models (no API key needed)
    if ":" in model or model.startswith(("llama", "mistral", "gemma", "qwen", "phi")):
        return OllamaClient(model)
    # Cloud API models
    elif model.startswith("gpt"):
        return OpenAIClient(config["openai_api_key"], model)
    elif model.startswith("claude"):
        return AnthropicClient(config["anthropic_api_key"], model)
    elif model.startswith("gemini") or model.startswith("models/"):
        return GeminiClient(config["gemini_api_key"], model)
    else:
        raise ValueError(f"Unsupported model: {model}")
