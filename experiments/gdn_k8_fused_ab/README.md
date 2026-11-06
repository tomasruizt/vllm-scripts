# Fused GDN MTP decode: K=8 support and split-V (vLLM PR #65874)

A/B of the fused GDN MTP decode CUDA kernel on main vs the PR, for Qwen3.6-35B-A3B with DSpark K=7/K=8 and native MTP3.
Host CPU load is sampled during every server session (`benchmarking/host_load.sh`), and `summarize.py` drops cells measured under CPU contention.
main is 5225a1f17f (2026-11-04), and the PR worktrees are built from the fork branch on top of it.

## Files

* `setup.sh`: main with precompiled wheels plus PR worktrees with `_C_stable_libtorch` built from source. Defaults are for the H100, the header has the B300 settings.
* `run.sh`, `benchmark.yaml`: one server session per variant (`{main,pr,splitv,splitv2,...}-{k7,k8,mtp3}`), AIPerf over repeats and concurrencies, then GSM8K.
* `validate.sh`, `kernel_bench.py`: GDN kernel tests, compute-sanitizer and the kernel microbenchmark (CUDA vs Triton, `--tokens` per request).
* `ncu.sh`, `ncu_profile.py` (`make ncu`): NCU full set of the fused kernel per build, e.g. `NCU_ARGS="--batch 16 --tokens 4"`. Pass `--clock-control none --cache-control none` via `NCU_FLAGS` to match the microbenchmark's boost clocks and warm caches.
* `kernel_variants.sh`: A/B of kernel source variants (`v-<name>.cu`) in one scratch worktree with incremental rebuilds, e.g. to attribute a regression to one source change.
* `split_sweep.sh`: forced split-V levels on a tuning build, which set the `choose_split_v` threshold.
* `crossover_bench.py`: recurrent vs chunked GDN kernels as a function of tokens per request.
* `summarize.py`, `plot_pareto.py`: long-format parquet of all AIPerf cells, and the throughput-interactivity curve.

## Run

```bash
make launch RUN_ID=gdn-mtp3-ab-20261106-1 VARIANTS="main-mtp3 splitv3-mtp3" \
  EXTRA_ENV="MAX_NUM_SEQS=256 CONCURRENCIES=1,2,4,8,16,32,64,128,256"
make validate RUN_ID=gdn-mtp3-validate-20261106-1 LABELS="splitv2 splitv3" BENCH_LABELS="main splitv2 splitv3" BENCH_TOKENS="4 9"
make fetch-results summarize RUN_ID=...
```

`REMOTE=RH-nm-automation-h100-standalone-1-preserve` switches to the H100 (FP8 model, its own paths).
