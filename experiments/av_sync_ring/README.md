# AV confidence copy: where to wait for it

Question: main waits for the previous step's confidence copy in `record_confidences` at every step (default 2-slot ring). Does that cost TTFT, and is it worth moving the wait to `get_num_tokens` (vllm#63355, lag stays 1) or using a 3-slot ring (wait on a copy that is 2 steps old, lag 2)?

Answer (2026-11-25): the wait costs 1.3 ms of prefill in a synthetic single-request test with long prompts, and both fixes remove it. With real prompts, closed loop or Poisson arrivals, the gain is inside the noise (±1 ms). vllm#63355 was closed on this evidence.

Setup: vLLM main `57906d3cc0` (2026-11-25), Qwen3-4B + `RedHatAI/Qwen3-4B-speculator.dspark` K=5, 1x H100, temperature 0, prefix caching off. Every variant ran on 2 GPUs in opposite order ([launch.sh](launch.sh)). TTFT and prefill are server-side (`vllm:time_to_first_token_seconds`, `vllm:request_prefill_time_seconds`).

Variants:

- `k5`: fixed K.
- `av_main`: AV on main.
- `av_depth3`: AV on main with `--async-scheduling-depth 3`.
- `av_ring3`: AV on main with `max(3, max_concurrent_batches)` ring slots.
- `av_lazy`: vllm#63355 merged with main (`9a929dfe45`).

## Synthetic: 8,200 input tokens, 16 output tokens, c=1

30 s, 2 trials per GPU (160 requests each).

| variant | prefill (ms) | vs fixed K | AL | drafts verified |
|---|---|---|---|---|
| `k5` | 138.3 | | 3.64 | 100% |
| `av_main` | 139.5 | +1.3 | 3.27 | 71.2% |
| `av_depth3` | 140.2 | +2.0 | 3.40 | 77.4% |
| `av_ring3` | 138.3 | 0.0 | 3.41 | 77.6% |
| `av_lazy` | 138.2 | 0.0 | 3.28 | 71.3% |

- The gap is the same on both GPUs (`av_main` +1.2 and +1.4 ms, trials within 0.3 ms).
- The 3-slot ring also changes the budget: the 1.0 fill of a new request lasts 2 steps, so short outputs verify more (71% to 78%). The lazy wait keeps main's behavior.
- The lazy wait's decode time per request is 0.35 to 0.5 ms above main in all 4 cells, so about a third of the prefill gain moves into decode. It is a one-time cost, not per step (next section).

## Synthetic decode

- 512 in, 512 out, c=1, 2 trials per GPU: time per decode step is 5.66 ms for `av_main`, `av_lazy` and `av_ring3` alike (cells within 0.02 ms), output throughput 468, 468 and 466 tok/s.
- 512 in, 256 out, c=16, 1 trial per GPU: output throughput within ±1.5% across all variants (5,942 to 6,062 tok/s).

## Real prompts

Mixed `RedHatAI/speculator_benchmarks` prompts (9 subsets, seeded shuffle, same requests for every variant), up to 1,024 output tokens, `--max-num-seqs 128`. Server TTFT in ms, and `av_lazy` minus `av_main` per GPU.

| load | requests per cell | `k5` | `av_main` | `av_lazy` | lazy - main, per GPU |
|---|---|---|---|---|---|
| closed loop c=1 | 32 | 19.0 | 19.9 | 19.3 | -0.95, -0.19 |
| closed loop c=8 | 64 | 28.3 | 28.6 | 28.4 | -0.88, +0.49 |
| closed loop c=32 | 256 | 38.9 | 40.1 | 41.9 | -3.24, +6.86 |
| closed loop c=128 | 1,024 | 100.5 | 100.3 | 98.4 | -1.12, -2.60 |
| Poisson 4 req/s | 240 (2 trials) | 23.3 | 21.6 | 22.5 | +0.68, +1.14 |
| Poisson 12 req/s | 720 (2 trials) | 25.4 | 26.3 | 26.4 | +0.31, -0.20 |
| Poisson 24 req/s | 1,440 (2 trials) | 69.0 | 42.5 | 42.8 | -0.02, +0.65 |

- No consistent TTFT gain: the sign flips between GPUs or sits within ±1 ms. Output throughput of `av_lazy` is within ±1.3% of `av_main` in every row.
- Side finding, closed loop: AV is slower than fixed K at low and medium concurrency, because it trims verification that would have paid off.

| closed loop | `k5` tok/s | `av_main` tok/s | AV vs fixed K | drafts verified |
|---|---|---|---|---|
| c=1 | 647 | 595 | -8% | 73% |
| c=8 | 4,086 | 3,369 | -18% | 49% |
| c=32 | 12,926 | 10,814 | -16% | 64% |
| c=128 | 20,142 | 20,418 | +1% | 94% |

## Commands

```
vllm serve Qwen/Qwen3-4B --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --speculative-config '{"model":"RedHatAI/Qwen3-4B-speculator.dspark","method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}'

# synthetic prefill
aiperf profile --model Qwen/Qwen3-4B --endpoint-type chat --streaming --synthetic-input-tokens-mean 8192 --synthetic-input-tokens-stddev 0 --output-tokens-mean 16 --output-tokens-stddev 0 --extra-inputs ignore_eos:true --extra-inputs temperature:0 --use-server-token-count --concurrency 1 --benchmark-duration 30 --warmup-request-count 3 --server-metrics http://127.0.0.1:8860/metrics

# real prompts, closed loop (c=32) and Poisson (12 req/s)
aiperf profile --model Qwen/Qwen3-4B --endpoint-type chat --streaming --input-file prompts/mixed.jsonl --custom-dataset-type single_turn --dataset-sampling-strategy sequential --extra-inputs max_tokens:1024 --extra-inputs temperature:0 --use-server-token-count --concurrency 32 --request-count 256 --warmup-request-count 32 --server-metrics http://127.0.0.1:8860/metrics
aiperf profile ... --request-rate 12 --arrival-pattern poisson --request-count 720 --warmup-request-count 60
```

## Run

`make launch watch fetch-results analyze RUN_ID=<new id>`, with `WORKLOADS` (`prefill`, `decode`, `decode_c1`, `chat_c<N>`, `chat_r<rate>`), `VARIANTS` and `MAX_NUM_SEQS` as needed. [analyze.py](analyze.py) writes `results/<run>/metrics.parquet` (long format) and prints the tables.

The host needs the variant trees under `~/code/av-sync`: `main` (vLLM checkout with its venv), and `ring3` and `lazy` as copies of `main/vllm` with the patched `adaptive_verification.py`. They are selected through `PYTHONPATH`.

Not run: a larger model. A Qwen3.5-27B TP2 run (`MODEL_KEY=27b`) was started and stopped before it produced results.
