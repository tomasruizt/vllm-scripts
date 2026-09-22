# Fused GDN execution confirmed

- [PR 52539](https://github.com/vllm-project/vllm/pull/52539), commit `cdb8545a91`, expanded PR 51674's ratio-8 kernel to ratios 1/2/3/4/8. Both benchmark builds include it; their GDN Python implementation files are byte-identical.
- Separate vLLM 0.30.0 probes used the pinned checkpoints, FP8 weights, BF16 Mamba states, MRV2, seven proposals, and the benchmark CUDA-graph sizes. PyTorch CUDA traces captured generation after warmup, with batches 1 and 32 and 64 output tokens each.

| Model | Value/key head ratio | Fused CUDA launches |
| --- | ---: | ---: |
| Qwen3.5-4B | 2 | 816 |
| Qwen3.5-27B | 3 | 1,392 |
| Qwen3.6-35B-A3B | 2 | 1,110 |

- All traces contain `gdn_decode_post_conv_mtp_kernel<__nv_bfloat16, ratio, false>` during generation. [Kernel names, counts, and trace paths](kernel-probes/summary.json).
- This confirms actual execution in the release probes, not just static eligibility. PR 52297 was not independently profiled; it shares the GDN implementation and release kernels. Mixed/prefill batches can still use fallback paths.
- These diagnostic timings are excluded from the benchmark tables. The initial Nsight attempt stalled; the completed evidence comes from PyTorch CUDA profiling.
