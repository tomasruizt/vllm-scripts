# Small DeepSeek AV prefill profile

Measure long-prompt TTFT and capture Torch profiles with a small DeepSeek-V4-Flash-DSpark clone and dummy weights.
This isolates prefill and synchronization overhead, it does not measure model accuracy or representative speculative acceptance.
The workload uses 8,192 random input tokens, 16 output tokens and concurrency 1.
With dummy weights, client TTFT can include delayed delivery of empty decoded text. Use server-side TTFT when interpreting prefill cost.

## Variants and external prerequisites

| Variant | Required vLLM setup |
| --- | --- |
| `nospec` | Base checkout, no speculative decoding. |
| `k5` | Base checkout, DSpark K=5. |
| `k5av` | Base checkout, DSpark K=5 with adaptive verification. |
| `k5av_nosync` | External diagnostic patch honoring `VLLM_DIAG_SKIP_AV_CONF_SYNC=1` to skip the confidence-copy event synchronization. This patch is not included here and the environment variable alone does not implement it. |
| `k5av_lazysync` | External checkout at `~/code/vllm-av-lazy-sync`, selected through `PYTHONPATH`, containing the lazy confidence synchronization fix. That checkout and its patch are not included here. |

The base interpreter is `~/code/vllm-dsv41/.venv/bin/python`, and AIPerf is expected at `~/aiperf-venv/bin/aiperf`.
The pinned clone snapshot and shared cache location are recorded in `run.py` and `launch.sh`.
Record the external checkout commits and patches with any new diagnostic results before comparing them.

## Run

The default launcher runs `nospec`, `k5` and `k5av`, each under a separate one-GPU `chg run` reservation.
Use a new run ID for each attempt:

```bash
make launch RUN_ID=dsv4-small-av-prefill-new-run
make watch RUN_ID=dsv4-small-av-prefill-new-run
```

To select diagnostic variants, invoke the launcher on the GPU host with `ROOT` pointing to a fresh run directory and `VARIANTS=k5av,k5av_nosync,k5av_lazysync` after installing the external patches.
The Makefile launch target copies only `run.py` and `launch.sh`, it does not copy or install those patches.
Keep fetched logs and profiles in the ignored `results/` directory.
