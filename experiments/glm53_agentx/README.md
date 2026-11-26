# GLM-5.3-Flash on AgentX with vLLM

AgentX is the agentic benchmark of SemiAnalysis InferenceX: it replays recorded coding-agent sessions (very long, mostly cached prompts) on a fixed number of session lanes. This experiment runs it against vLLM with the settings of the published SGLang recipe for GLM-5.3-Flash, to get comparable numbers.

Status: preliminary. One run per point, and the c32 point is under investigation (see below). The profiler trial on the same setup is in [../nsight_perf_sdk](../nsight_perf_sdk).

## Setup

- Hardware: 4x B200 (183 GiB), TP4. The published SGLang numbers are from a GB300 tray with 4 GPUs.
- vLLM `0.34.1rc1.dev173+g5aea30579` (nightly image from 2026-11-26).
- `zai-org/GLM-5.3-Flash` at revision `46a050bec1`, native FP8 weights, FP8 KV cache, prefix caching on.
- Speculative decoding: MTP with 5 speculative tokens. AgentX prescribes a fixed ("golden") acceptance length per model, 3.60 for this setup, which vLLM applies with `rejection_sample_method=synthetic`.
- Host KV offloading only where the SGLang recipe enables HiCache (from c40 on): `SimpleCPUOffloadConnector`, 88 GiB per rank, eager.
- Client: agentx-harness 1.0.14, dataset `semianalysis_cc_traces_weka_062126`, 3,600 s per point, 10 warmup requests per lane, server restarted for every point.

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve zai-org/GLM-5.3-Flash \
    --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c \
    --tokenizer-mode glm5_next --tensor-parallel-size 4 --kv-cache-dtype fp8 --language-model-only \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.60}' \
    --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice
# c48 only:
    --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":94489280512,"lazy_offload":false}}'

agentx profile --url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
    --model zai-org/GLM-5.3-Flash --tokenizer zai-org/GLM-5.3-Flash --tokenizer-trust-remote-code \
    --concurrency $C --benchmark-duration 3600 --warmup-requests-per-lane 10 \
    --public-dataset semianalysis_cc_traces_weka_062126 --server-metrics http://127.0.0.1:8000/metrics
```

## Results

Throughput is per GPU and counts input plus output tokens, as InferenceX reports it. Interactivity is 1000 / mean ITL. AIPerf's ITL is per token, not per streamed chunk, so it already accounts for several tokens arriving per step. The SGLang columns are the public InferenceX numbers (GB300, SGLang, published 2026-11-06, [reference/semianalysis-sglang-fp8.csv](reference/semianalysis-sglang-fp8.csv)).

| Concurrency | vLLM tok/s/GPU | SGLang tok/s/GPU | vLLM interactivity | SGLang interactivity | vLLM TTFT (ms) | SGLang TTFT (ms) | vLLM cache hit | vLLM req/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 7,260 | 5,860 | 535 | 407 | 189 | 810 | 98.0% | 0.09 |
| 16 | 29,094 | 27,280 | 351 | 225 | 212 | 600 | 96.2% | 0.75 |
| 32 | 44,537 | 52,940 | 95 | 125 | 1,927 | 868 | 85.8% | 1.10 |
| 48 (host offload) | 69,272 | 63,873 | 115 | 89 | 1,194 | 1,166 | 93.0% | 1.80 |

- vLLM is ahead on throughput and interactivity at c1, c16 and c48, and behind at c32.
- At c48, 63.8% of the prompt tokens are served from the GPU KV cache and 29.2% from host memory.
- 2 or 3 requests fail per run (out of 322 to 6,521): empty responses and broken pipes. Not investigated.
- The mean input length matches the public runs within a few percent at c16 and above (about 155k tokens), which shows that a 60-minute window replays the same sessions. Shorter runs (20 min) cover different sessions, so their throughput is not comparable.

`make summary` rebuilds this table (and `results/metrics.parquet`), `make plot` draws the throughput-interactivity curve.

## The c32 point

The c32 run is slower than the c48 run from the first minutes on, although c48 has more requests decoding at the same time (16 against 11 on average). A curve that improves with load points to something specific to that run, and two reruns are in progress (plain repeat, and c32 with host offloading).

What we know so far:

- The only configuration difference between c32 and c48 is host offloading. Without it, the GPU KV cache got tight at c32 (peak usage 83%) and the cache hit rate dropped to 85.8%.
- Uncached prompt tokens are expensive for everyone: a step that carries a prefill chunk delays all requests decoding in that step (see the microbenchmark below). At c32 about 25k uncached tokens arrive per second, against a prefill speed of about 52k tok/s, so roughly half of the wall time goes to steps with prefill. At c16 it is under 10%.
- The batch-size cap of the specialized decode (see below) is not the main cause: it is worth about 15% per step, and c48 exceeds the cap more often than c32.

## Step-time microbenchmark

`forward.sh` times single steps with `vllm bench forward` (TP4, real weights, MTP 5, step including drafter and sampler, 30k tokens of context per request):

| Step | Median (ms) |
|---|---|
| 1 request | 7.9 |
| 4 requests | 11.0 |
| 8 requests | 13.4 |
| 10 requests | 14.8 |
| 11 requests | 15.2 |
| 16 requests | 16.8 |
| 24 requests | 18.5 |
| 32 requests | 21.2 |
| 8 requests + 2k-token prefill chunk | 54.7 |
| 8 requests + 8k-token prefill chunk | 163.6 |
| 8k-token prefill chunk alone | 157.6 |

`vllm bench forward` does not engage GLM-5.3-Flash's specialized decode (`decode_backend=latency_specialized`), so these are stock-path times. In serving, the specialized path is active by default on this setup. It covers decode-only steps of up to 64 rows (10 requests with MTP 5) and falls back to the stock path beyond that or when a step contains prompt rows. At c1 a serving step takes about 6.7 ms (1.87 ms ITL x 3.6 accepted tokens) against 7.9 ms here.

## Running it

The scripts target a Kubernetes or OpenShift node with free Blackwell GPUs and a node-local storage class. `make` variables select cluster, namespace, node and resource prefix (see the [Makefile](Makefile)).

```bash
make box prepare                       # model disk, CPU pod, downloads (weights 306 GiB, client, traces)
make run CONC=16 DURATION=3600 WARMUP=10 RUN_ID=full-c16-1
make run CONC=48 DURATION=3600 WARMUP=10 RUN_ID=full-c48-1 OFFLOAD=88 MEM=800Gi
make run ENTRY=forward.sh RUN_ID=forward-1                                    # step-time microbenchmark
make fetch-results RUN_ID=full-c16-1   # into results/ (git-ignored)
make summary plot
```

Smoke-test script changes on one GPU with the small clone before spending 4 GPUs: `make run ENTRY=forward.sh GPUS=1 MEM=128Gi MODEL=inference-optimization/GLM-5.3-Flash-2.6B-A2.3B-small REVISION= LOAD_FORMAT=dummy RUN_ID=forward-smoke-1`.

## Caveats

- Different GPUs: B200 here, GB300 for the SGLang numbers.
- One run per point, on a shared node. Host CPU load was not recorded for these runs (`run.sh` records it from now on).
- An older nightly (`0.34.0rc1.dev776+g7c3a9302f`, 2026-11-18) crashed once at c1 with a CUDA "index out of bounds" assert in the KDA layer after 15 minutes. Not reproduced on the nightly used here (one run).
