# GLM-5.3-Flash on AgentX with vLLM

AgentX is the agentic benchmark of SemiAnalysis InferenceX: it replays recorded coding-agent sessions (very long, mostly cached prompts) on a fixed number of session lanes. This experiment runs it against vLLM with the settings of the published SGLang recipe for GLM-5.3-Flash, to get comparable numbers.

Status: preliminary. One run per point (two at c32 without offloading). With host KV offloading on, vLLM is ahead of the published SGLang numbers at every measured point. Without it, vLLM falls behind at c32 (see "The c32 point").

## Setup

- Hardware: 4x B200 (183 GiB), TP4. The published SGLang numbers are from a GB300 tray with 4 GPUs.
- vLLM `0.34.1rc1.dev173+g5aea30579` (nightly image from 2026-11-26).
- `zai-org/GLM-5.3-Flash` at revision `46a050bec1`, native FP8 weights, FP8 KV cache, prefix caching on.
- Speculative decoding: MTP with 5 speculative tokens. AgentX prescribes a fixed ("golden") acceptance length per model, 3.60 for this setup, which vLLM applies with `rejection_sample_method=synthetic`.
- Host KV offloading: every point up to c32 is measured without and with it, c48 only with it (the SGLang recipe enables HiCache from c40 on). `SimpleCPUOffloadConnector`, 88 GiB per rank, eager.
- Client: agentx-harness 1.0.14, dataset `semianalysis_cc_traces_weka_062126`, 3,600 s per point, 10 warmup requests per lane, server restarted for every point.

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve zai-org/GLM-5.3-Flash \
    --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c \
    --tokenizer-mode glm5_next --tensor-parallel-size 4 --kv-cache-dtype fp8 --language-model-only \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.60}' \
    --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice
# runs with host offload only:
    --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":94489280512,"lazy_offload":false}}'

agentx profile --url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
    --model zai-org/GLM-5.3-Flash --tokenizer zai-org/GLM-5.3-Flash --tokenizer-trust-remote-code \
    --concurrency $C --benchmark-duration 3600 --warmup-requests-per-lane 10 \
    --public-dataset semianalysis_cc_traces_weka_062126 --server-metrics http://127.0.0.1:8000/metrics
```

## Results

Throughput is per GPU and counts input plus output tokens, as InferenceX reports it. Interactivity is 1000 / mean ITL. AIPerf's ITL is per token, not per streamed chunk, so it already accounts for several tokens arriving per step. The SGLang columns are the public InferenceX numbers (GB300, SGLang, published 2026-11-06, [reference/semianalysis-sglang-fp8.csv](reference/semianalysis-sglang-fp8.csv)).

"Off" and "on" are vLLM without and with host KV offloading. SGLang runs with HiCache from c40 on.

| Concurrency | tok/s/GPU off | tok/s/GPU on | tok/s/GPU SGLang | Interactivity off | Interactivity on | Interactivity SGLang | TTFT off (ms) | TTFT on (ms) | TTFT SGLang (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7,260 | 7,262 | 5,860 | 534 | 533 | 407 | 189 | 188 | 810 |
| 16 | 29,094 | 29,124 | 27,280 | 351 | 352 | 225 | 212 | 181 | 600 |
| 24 | 44,395 | 46,244 | 41,907 | 219 | 279 | 160 | 635 | 257 | 768 |
| 32 | 45,227 (2 runs) | 59,891 | 52,940 | 102 | 220 | 125 | 1,911 | 379 | 868 |
| 48 | | 69,272 | 63,873 | | 115 | 89 | | 1,194 | 1,166 |

- With host offloading on, vLLM is ahead on throughput and interactivity at every point: +7% to +24% throughput and 1.3x to 1.8x the interactivity.
- Host offloading costs nothing at low load: c1 and c16 are identical with and without it. So it can simply stay on, which matters because it is a server startup setting.
- Without it, vLLM stops scaling between c24 and c32 (44.4k and 45.2k tok/s/GPU) and falls behind SGLang at c32.
- Cache hits from host memory: 0.4% of the prompt tokens at c16, 3.8% at c24, 7.2% at c32 and 29.2% at c48.
- 1 to 3 requests fail per run (out of 322 to 6,521): empty responses and broken pipes. Not investigated.
- The mean input length matches the public runs within a few percent at c16 and above (about 155k tokens), which shows that a 60-minute window replays the same sessions. Shorter runs (20 min) cover different sessions, so their throughput is not comparable.

`make summary` rebuilds this table (and `results/metrics.parquet`), `make plot` draws the throughput-interactivity curve.

## The c32 point

Without host offloading, c32 is slower than c48 with it, although c48 has more requests decoding at the same time (16 against 11 on average). A plain repeat reproduced the first run within 3% on throughput (ITL 9.2 against 10.5 ms), so it is not a disturbed run. Turning host offloading on at c32 changes the picture: throughput +32%, ITL from about 10 ms to 4.6 ms, TTFT from 1.9 s to 0.4 s.

The cause is eviction of cached prefixes. The AgentX client reports how many prompt tokens could have been cache hits given the traces, next to what the server served from cache:

| Run | Theoretical hit | Hit from GPU KV cache | Hit from host memory | Lost |
|---|---|---|---|---|
| c16 | 97.5% | 96.2% | off | 1.3% |
| c24 | 97.8% | 93.0% | off | 4.8% |
| c24, host offload | 97.8% | 92.7% | 3.8% | 1.3% |
| c32, run 1 | 97.7% | 85.8% | off | 11.9% |
| c32, run 2 | 97.7% | 86.8% | off | 10.9% |
| c32, host offload | 97.9% | 89.2% | 7.2% | 1.5% |
| c48, host offload | 97.4% | 63.8% | 29.2% | 4.4% |

Why offloading matters this much:

- The KV cache is 88 GiB per GPU (the server uses 82.6 of 178 GiB for weights and runtime). 32 sessions with about 160k tokens of context each want their prefix kept while they idle between turns, and the blocks held by running requests alone peak at 81 to 83% of the pool.
- `vllm:kv_cache_usage_perc` only counts blocks held by running requests. The prefix of an idle session lives in blocks that count as free, and it is overwritten when running requests need the space. So eviction starts long before the gauge reads 100% (its mean at c32 is 32%).
- An evicted session has to prefill up to 160k tokens again on its next turn, which keeps more requests in flight (10 on average against 6.1 with offloading) and raises the pressure further. Offloading breaks that loop: only 7.2% of the prompt tokens come from host memory, but the GPU cache hit rate itself also rises from 86% to 89%.
- Uncached prompt tokens are expensive for everyone: a step that carries a prefill chunk delays all requests decoding in that step (see the microbenchmark below). At 86% hits about 25k uncached tokens arrive per second, against a prefill speed of about 52k tok/s, so roughly half of the wall time goes to steps with prefill. At c16 it is under 10%.
- The batch-size cap of the specialized decode (see below) is not the main cause: it is worth about 15% per step, and c48 exceeds the cap more often than c32.

Caveats: the offloading run and the repeat ran at the same time on separate GPUs of the same node, and the SGLang numbers at c32 are without HiCache, so the offloading row is not a like-for-like comparison. The two setups also differ in GPU memory (B200 against GB300), which may be why the KV cache gets tight earlier here. Not verified.

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
- One run per point, on a shared node. Host CPU load is recorded for the later runs only (c24, and c1 and c16 with offloading): 8 to 17 busy cores on average out of 224.
- An older nightly (`0.34.0rc1.dev776+g7c3a9302f`, 2026-11-18) crashed once at c1 with a CUDA "index out of bounds" assert in the KDA layer after 15 minutes. Not reproduced on the nightly used here (one run).
