# Benchmark Reproduction Guide

This document contains all commands used to run the benchmarks for PR #33318.

**Results Directory**: `~/code/vllm-scripts/drafter-supports-multiple-kvcache-groups/`

## Environment Setup

```bash
# Activate virtual environment
source /home/shadeform/.venv/bin/activate

# Set HuggingFace token
export HF_TOKEN=<censored>

# Results directory
RESULTS_DIR=~/code/vllm-scripts/drafter-supports-multiple-kvcache-groups
```

## Common Benchmark Parameters

All benchmarks use:
- **Temperature**: 0.0 (greedy decoding)
- **Top-p**: 1.0
- **Dataset**: philschmid/mt-bench (test split)
- **Prefix caching**: Disabled
- **Backend**: openai-chat
- **Endpoint**: /v1/chat/completions

---

## Gemma-3 Benchmarks (Multi-KV Cache Showcase)

> **Note**: Gemma-3 multimodal models (4b, 12b, 27b) have torch.compile bugs. Only text-only models (270m, 1b) work.

### Configuration
- **Target**: google/gemma-3-1b-it (text-only, has sliding + full attention)
- **Draft**: google/gemma-3-270m-it (text-only, has sliding + full attention)
- **K (num_speculative_tokens)**: 3

### Baseline Server

```bash
vllm serve google/gemma-3-1b-it \
  --port 8000 \
  --no-enable-prefix-caching
```

### Spec Decode Server

```bash
vllm serve google/gemma-3-1b-it \
  --port 8000 \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "draft_model", "model": "google/gemma-3-270m-it", "num_speculative_tokens": 3}'
```

### Benchmark Command

```bash
vllm bench serve \
  --model google/gemma-3-1b-it \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --hf-split test \
  --num-prompts 50 \
  --max-concurrency 1 \
  --request-rate 1 \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --temperature 0.0 \
  --top-p 1.0 \
  --save-result \
  --result-filename $RESULTS_DIR/results/gemma3-1b-{config}-c1-run{N}.json
```

---

## GPT-OSS Benchmarks (MoE Multi-KV Cache Showcase)

### Configuration
- **Target**: openai/gpt-oss-120b (MoE model)
- **Draft**: openai/gpt-oss-20b (MoE model)
- **K (num_speculative_tokens)**: 3
- **Prefix caching**: Disabled

### Baseline Server (No Speculative Decoding)

```bash
vllm serve openai/gpt-oss-120b \
  --port 8000 \
  --no-enable-prefix-caching
```

### Spec Decode Server

```bash
vllm serve openai/gpt-oss-120b \
  --port 8000 \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "draft_model", "model": "openai/gpt-oss-20b", "num_speculative_tokens": 3}'
```

### Benchmark Command

```bash
vllm bench serve \
  --model openai/gpt-oss-120b \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --hf-split test \
  --num-prompts 50 \
  --max-concurrency 1 \
  --request-rate 1 \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --temperature 0.0 \
  --top-p 1.0 \
  --save-result \
  --result-filename ~/code/vllm-scripts/drafter-supports-multiple-kvcache-groups/results/gpt-oss-120b-{config}-c1-run{N}.json
```

---

## Verification Benchmarks (Qwen3)

### Configuration
- **Target**: Qwen/Qwen3-32B
- **Draft**: Qwen/Qwen3-1.7B
- **K (num_speculative_tokens)**: 3
- **Prefix caching**: Disabled

### Baseline Server

```bash
vllm serve Qwen/Qwen3-32B \
  --port 8000 \
  --no-enable-prefix-caching
```

### Spec Decode Server

```bash
vllm serve Qwen/Qwen3-32B \
  --port 8000 \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "draft_model", "model": "Qwen/Qwen3-1.7B", "num_speculative_tokens": 3}'
```

---

## Verification Benchmarks (Llama-3.1-8B + EAGLE3)

### Configuration
- **Target**: meta-llama/Llama-3.1-8B-Instruct
- **Drafter**: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
- **Prefix caching**: Disabled

### Baseline Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --no-enable-prefix-caching
```

### EAGLE3 Spec Decode Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 3}'
```

### Benchmark Command

```bash
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 50 \
  --max-concurrency 1 \
  --request-rate 1 \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --temperature 0.0 \
  --top-p 1.0
```

---

## Notes

- **Prefix caching** is disabled (`--no-enable-prefix-caching`) to ensure fair benchmark comparisons across runs
- All benchmarks use the same dataset: `philschmid/mt-bench` (test split, 50 prompts)
- Concurrency: 1, Request rate: 1 RPS
- Hardware: NVIDIA H200 (143 GiB)
