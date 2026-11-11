# Top-k/top-p CUDA filter: E2E A/B with spec decoding

AIPerf A/B of the CUDA top-k/top-p masking kernel (vLLM PR branch `perf/topk-topp-cuda-filter`) vs main's Triton kernel, on Qwen3.6-35B-A3B with DSpark K=7 and the model card's sampling (T=0.7, top-p 0.8, top-k 20, presence penalty 1.5).
Spec-decode verification masks every target row through `apply_top_k_top_p`, so it always runs the kernel, while plain decoding uses the fused FlashInfer sampler.
Copied from `../gdn_k8_fused_ab` (same runner, summary and plot), with the variants pointing at the `topk-cuda-main` and `topk-cuda-pr2` trees built by `../topk_topp_microbench/pr_build.sh`.

```bash
make launch RUN_ID=topk-topp-e2e-20261106-1            # main-k7 and pr-k7 in parallel, one GPU each
make watch fetch-results summarize RUN_ID=topk-topp-e2e-20261106-1
```
