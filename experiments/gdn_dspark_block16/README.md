# GDN DSpark block-16 validation

Validate fixed K=15 and adaptive verification against a no-speculation baseline for vLLM PR #60229, using AIPerf SPEED-Bench, GSM8K, execution metadata checks and c1 CUDA-event/Nsight profiles.
Timed runs do not install the diagnostic execution probes.

## Run variants

| Entry point | Model and source | Purpose |
| --- | --- | --- |
| `run.py`, `prepare.sh` | Qwen3.6-27B + satgeze DSpark, local patched vLLM checkout | Original validation. The Makefile defaults identify the `gdn-dspark-block16-20261027-1203-tp2` analysis run. |
| `launch_radix.sh`, `profile_radix.sh` | Qwen3.8-27B + RadixArk DSpark, local patched vLLM checkout | Intermediate validation and c1 profiling. |
| `launch_author.sh` | Qwen3.8-27B + RadixArk DSpark, clean author commit `555978f8cf2cd21b22adba5d09d5f6e9df357714` (PR #60229 on 2026-10-27) | Latest author validation. Three methods run concurrently on separate B300 pairs, with a fourth pair for diagnostics and c1 profiles. |

The original and latest author runs use different source snapshots and execution arrangements. Their performance differences are not a controlled before/after comparison.
The author launcher checks the exact commit and clean source tree before running.

## Prerequisites and outputs

Launch on the GPU host with `chg` installed and the model snapshots available in the shared cache. The author launcher reserves eight GPUs through `chg run`.
Inspect each launcher before use, since source checkout paths, model paths, ports and GPU selections reflect the original machines.
`prepare.sh` downloads the original Qwen3.6 models, it does not prepare the newer Qwen3.8/RadixArk pair.
The runner accepts `SOURCE_DIR`, `RUN_ROOT`, `TARGET_MODEL_DIR`, `DRAFT_MODEL_DIR` and `TENSOR_PARALLEL_SIZE`, but launchers set their own variant-specific values.

Fetch and analyze the latest author run locally with:

```bash
make fetch-results RUN_ID=gdn-author-555978f8-20261027-2210-tp2
make analyze RUN_ID=gdn-author-555978f8-20261027-2210-tp2
make plot-pareto RUN_ID=gdn-author-555978f8-20261027-2210-tp2 PLOT_TITLE='Qwen3.8-27B · 2× B300, TP2'
```

Keep fetched artifacts and generated summaries under the ignored `results/`, `plots/` and `report/` directories.
