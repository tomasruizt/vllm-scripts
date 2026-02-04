# TODOs

## Showcase New Feature
This PR mainly enables the use of multi-kv cache models as drafters.
Therefore we added unit tests for speculative decoding for:
* gemma3 (270m), and
* gpt-oss-120b/20b (MoE)

I would like to show speedups for these cases, using the benchmark
`vllm bench serve`.

- Run target=google/gemma-3-27b-it, and draft=google/gemma-3-270m-it
- Run target=openai/gpt-oss-120b, and draft=openai/gpt-oss-20b

Dataset: `philschmid/mt-bench`, concurrency levels [1, 16], request-rate [1, 16]

Put raw logs in the ~/code/bench/ directory. Then also create a markdown summarizing the results. Orient yourself on the these benchmarks I have done in the past: https://tomasruizt.github.io/posts/06_vllm-spec-decode/


## Verify Performance Unaffected

Since we are touching the codebase, I would like to make sure the performance of draft_model and eagle-3 is not negatively affected. Please run these benchmarks, both in this branch and the `main` branch:

- For draft_model: target_model=Qwen/Qwen3-32B, draft_model=Qwen/Qwen3-1.7B
- For eagle-3: target_model="meta-llama/Llama-3.1-8B-Instruct", drafter=
"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


## Clarified Requirements

Based on discussion, the following additional requirements were clarified:

### 1. Baseline Comparisons
- Run each target model **without speculative decoding** to show actual speedup factors

### 2. Statistical Reliability
- Run each benchmark **2-3 times** to ensure consistent results

### 3. Metrics
- For **verification benchmarks**: metrics should stay unaffected (compare this branch vs main)
- For **new feature showcase**: metrics are completely novel (no prior baseline exists)

### 4. VLLM_BATCH_INVARIANT Testing
- Test gemma3 **with and without** `VLLM_BATCH_INVARIANT=1`
- Provide recommendation on whether it's needed

### 5. Memory Tracking
- Monitor GPU memory usage during benchmarks (new - never done before)

### 6. Speculative Tokens (K)
- Execute all tests with **K=3**
- Make recommendation if other K values would be interesting to explore

### 7. Eagle Drafter
- Use `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (EAGLE**3**, from test file)


## Issues Found

### Gemma3 Model Compatibility
- **gemma-3-27b-it**: vocab_size=262208, architecture=Gemma3ForConditionalGeneration (multimodal)
- **gemma-3-270m-it**: vocab_size=262144, architecture=Gemma3ForCausalLM (text-only)
- Vocab sizes differ by 64 tokens, architectures are different (multimodal vs text-only)
- Need to investigate if this causes compatibility issues for speculative decoding
