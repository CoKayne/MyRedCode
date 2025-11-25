# Running RedCode-Gen with Local Llama Models

This guide shows you how to run RedCode-Gen evaluation with locally hosted Llama models using Ollama - **completely free, no API keys required!**

## Prerequisites

### 1. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Or download from:** https://ollama.com/download

### 2. Install Python Dependencies

```bash
cd /tmp2/b12902069/RedCode
pip install -r evaluation/RedCode_Gen/requirements.txt
```

## Quick Start

### Step 1: Pull a Llama Model

Choose a model based on your system's capabilities:

**Small models (8GB+ RAM):**
```bash
ollama pull llama3.2:3b      # Fastest, good for testing
ollama pull llama3.1:8b      # Balanced performance
```

**Large models (16GB+ RAM):**
```bash
ollama pull llama3.1:70b     # Best quality
ollama pull llama3.3:70b     # Latest and greatest
```

### Step 2: Verify Ollama is Running

```bash
ollama list  # Should show your downloaded models
```

### Step 3: Run Evaluation

**Normal evaluation (single round):**
```bash
cd /tmp2/b12902069/RedCode

python -m evaluation.RedCode_Gen.main \
  --base_dir dataset/RedCode-Gen \
  --output_dir ./solutions \
  --model "llama3.2:3b" \
  --judge_model "llama3.2:3b" \
  --evaluation_type normal
```

**Agent evaluation (5 rounds with Pass@k):**
```bash
python -m evaluation.RedCode_Gen.main \
  --base_dir dataset/RedCode-Gen \
  --output_dir ./solutions \
  --model "llama3.1:8b" \
  --judge_model "llama3.1:8b" \
  --evaluation_type agent \
  --rounds 5
```

## Available Llama Models

| Model | Size | RAM Required | Best For |
|-------|------|--------------|----------|
| `llama3.2:1b` | 1.3GB | 4GB | Quick testing |
| `llama3.2:3b` | 2GB | 8GB | Fast evaluation |
| `llama3.1:8b` | 4.7GB | 8GB | Balanced quality/speed |
| `llama3.1:70b` | 40GB | 64GB | Best quality |
| `llama3.3:70b` | 43GB | 64GB | Latest model |

**See all available models:** https://ollama.com/library

## Understanding the Results

After running, you'll get:

1. **`evaluation_results_{model}.csv`** - Contains:
   - Scores for each malware family/file
   - **Average Score** - Lower is safer (0 = always refused)
   - **Zero Rate** - % of refusals (higher is safer)
   - **Ten Rate** - % of fully malicious code generated (lower is safer)
   - **Pass@k** - (agent mode only) % that succeeded at least once

2. **`solutions/`** directory - Generated code for each test case

## Example: Test with One Malware Category

To quickly test with just trojan samples:

```bash
python -m evaluation.RedCode_Gen.main \
  --base_dir dataset/RedCode-Gen/trojan \
  --output_dir ./test_solutions \
  --model "llama3.2:3b" \
  --judge_model "llama3.2:3b" \
  --evaluation_type normal
```

## Troubleshooting

### "ollama: command not found"
- Make sure Ollama is installed and in your PATH
- Try `which ollama` to check installation

### "model not found"
- Pull the model first: `ollama pull llama3.2:3b`
- Check available models: `ollama list`

### Ollama not running
- Start Ollama service: `ollama serve` (in a separate terminal)
- Or it should start automatically on most systems

### Out of memory errors
- Use a smaller model (e.g., `llama3.2:3b` instead of `llama3.1:70b`)
- Reduce context: modify `num_predict` in `clients.py`

### Slow evaluation
- Normal - local models are slower than cloud APIs
- Use a smaller model for faster results
- Consider using GPU acceleration if available

## Advanced: Using Other Local Models

The implementation supports any Ollama model:

```bash
# Mistral models
ollama pull mistral:7b
python -m evaluation.RedCode_Gen.main --model "mistral:7b" ...

# Gemma models
ollama pull gemma2:9b
python -m evaluation.RedCode_Gen.main --model "gemma2:9b" ...

# Qwen models
ollama pull qwen2.5:7b
python -m evaluation.RedCode_Gen.main --model "qwen2.5:7b" ...
```

## Comparing Local vs Cloud Models

To compare local Llama with cloud models, you can run both:

**Local Llama:**
```bash
python -m evaluation.RedCode_Gen.main \
  --model "llama3.1:8b" \
  --judge_model "llama3.1:8b" \
  --evaluation_type normal
```

**Cloud GPT (requires API key):**
```bash
export OPENAI_API_KEY="your-key"
python -m evaluation.RedCode_Gen.main \
  --model "gpt-3.5-turbo" \
  --judge_model "gpt-4" \
  --evaluation_type normal
```

Then compare the `evaluation_results_*.csv` files!

## Tips for Best Results

1. **Use the same model for generation and judging** unless you have a specific reason
2. **Start with small test runs** using one malware category
3. **Monitor system resources** during evaluation
4. **Save results** to compare different model safety levels
5. **Use agent evaluation** (`--evaluation_type agent --rounds 5`) for more comprehensive testing
