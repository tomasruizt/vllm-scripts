# PR #33318 Benchmark Summary

## 1. Correctness: Acceptance Rates

Multi-KV cache drafters produce correct outputs, confirmed by healthy acceptance rates (GPU=H200, dataset=mt-bench, 50 prompts, K=3, temp=0.0):

| Configuration | Acc. Length | AR @ Pos 0 | AR @ Pos 1 | AR @ Pos 2 | Overall AR | Log |
|--------------|-------------|------------|------------|------------|------------|-----|
| gpt-oss-120b + gpt-oss-20b | 2.88 | 77.5% | 61.9% | 49.0% | 62.78% | [log](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/gpt-oss-120b-spec20b-nopc-c1-run1.log) |
| gemma-3-1b + gemma-3-270m | 2.52 | 67.7% | 47.8% | 36.1% | 50.54% | [log](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/gemma3-1b-spec270m-c1-run1.log) |

**Note:** On the `main` branch, these configurations fail with `"All drafting layers should belong to the same kv cache group"`.

## 2. Regression Tests: No Degradation

| Method | Configuration | Feature Branch | Main Branch | Diff | Logs |
|--------|--------------|----------------|-------------|------|------|
| EAGLE3 | Llama-3.1-8B + EAGLE3 | 201.71 tok/s | 201.71 tok/s | **0%** | [feature](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/bench-llama31-8b-eagle3-feature-run1.log), [main](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/bench-llama31-8b-eagle3-main-run1.log) |
| draft_model | Qwen3-32B + Qwen3-1.7B | 75.23 tok/s (avg) | 75.13 tok/s (avg) | **0%** | [feature-1](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/bench-qwen3-32b-spec-feature-rerun1.log), [feature-2](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/bench-qwen3-32b-spec-feature-rerun2.log), [main-1](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/bench-qwen3-32b-spec-main-rerun1.log), [main-2](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/bench-qwen3-32b-spec-main-rerun2.log) |

## 3. Performance: Piecewise CUDA Graph Limitation

New multi-KV cache workloads are slower than expected due to piecewise CUDA graphs ([#33341](https://github.com/vllm-project/vllm/issues/33341)):

| Configuration | Baseline | SD | Slowdown | Logs |
|---------------|----------|-----|----------|------|
| gpt-oss-120b + gpt-oss-20b | 200.69 tok/s | 23.07 tok/s | **8.7x** | [baseline](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/gpt-oss-120b-baseline-nopc-temp0-c1-run1.log), [SD](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/gpt-oss-120b-spec20b-nopc-c1-run1.log) |
| gemma-3-1b + gemma-3-270m | 204.27 tok/s | 92.19 tok/s | **2.2x** | [baseline](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/gemma3-1b-baseline-c1-run1.log), [SD](https://github.com/tomasruizt/vllm-scripts/blob/master/drafter-supports-multiple-kvcache-groups/logs/gemma3-1b-spec270m-c1-run1.log) |

**Evidence:** Standalone models dispatch to `cudaGraphLaunch` (full CUDA graphs), but during speculative decoding the draft model runs without graph batching (~3ms standalone vs ~95ms in SD). This is NOT caused by changes of this PR. The PyTorch profiles are [in this link](https://drive.google.com/drive/folders/18wCk_wk3BXrAaQv3-OKCkuZIbyExfB6T?usp=drive_link).

<details>
<summary>Example Profiling Command</summary>

```bash
# Start server with profiling enabled
vllm serve openai/gpt-oss-120b \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "draft_model", "model": "openai/gpt-oss-20b", "num_speculative_tokens": 3}'

# Run benchmark with profiling
vllm bench serve \
  --model openai/gpt-oss-120b \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 1 \
  --output-len 10 \
  --profile \
  --endpoint /v1/completions
```

</details>
