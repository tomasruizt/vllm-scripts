## Qwen3.5-4B

| Engine | ITL without DFlash | ITL with DFlash | Toks/s DFlash |
|---|---:|---:|---:|
| vLLM | 3.29 ms | 8.80 ms | 569.9 |
| SGLang | 3.26 ms | 5.99 ms | 829.2 |

## Qwen3.5-27B

| Engine | ITL without DFlash | ITL with DFlash | Toks/s DFlash |
|---|---:|---:|---:|
| vLLM | 12.32 ms | 20.06 ms | 318.2 |
| SGLang | 11.91 ms | 17.93 ms | 364.5 |

ITL is the median interval between streamed chunks, not TPOT.
All runs used concurrency 1, 200 measured GSM8K requests after 20 warmups, 256 requested output tokens, FP8 target weights, and one H100 per engine, without profiling.
