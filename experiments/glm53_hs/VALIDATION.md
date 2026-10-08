# How we check hidden-state extraction

Hidden states are the intermediate numbers a model computes for each input token.
The test asks whether a Speculators client receives a readable file containing those numbers, with the right tokens and layers.

We use Speculators' actual [launcher](/home/tomasruizt/code/speculators/scripts/launch_vllm.py) and [client helper](/home/tomasruizt/code/speculators/src/speculators/data_generation/vllm_client.py), from commit `e3041ddb`.
The helper sends token IDs to `/v1/completions` with `max_tokens=1` and `return_token_ids=True`, checks the returned token IDs, and obtains the file path.
We wait for file writing to finish and run Speculators' own file checks.
The launcher adds the final layer, so these tests extract layers 5, 22, 43, and 45.

The CUDA-graph test has two stages:

1. Send 20 requests without copying reference tensors. Check that the files load, contain the requested tokens, and have finite values and four layers.
2. Send another 20 requests, copying the model's output before extraction stores it. Compare each saved file with copies from that same execution, requiring exact bytes and every token exactly once in order. Request/token labels come from the runner, independently of storage addresses.

We count actual graph replays and require every reference chunk to have used them.
We supply capture sizes from 8 to 512 tokens to cover these batches.
Deliberately corrupted values, swapped layers, and shifted tokens must fail the comparison.
Cases cover 8–1,921 tokens, processing and storage boundaries, mixed-length batches, repeated requests, reused cache space, and identical prompts.

**We do not compare separate executions for equality.**
Batching can change the model's numbers during normal operation.
Exact equality is appropriate only between a computed value and its saved copy.

Run from this directory, using an unused output folder:

```bash
HS_OUTPUT_DIR="$HOME/benchmarks/glm-hs-validation/recheck-default-v2" bash run-api.sh 1
```

This uses MRV2 with the fallback fix in `f95f7f102` and leaves graph-mode selection automatic.
The script fails if no graphs replay, even if the client requests succeed.
`run-api-piecewise.sh 1` reproduces the earlier run that explicitly selected piecewise graphs.

Scope: full shared FP8 weights on four B300 GPUs, text input, prompt states, and prefix caching disabled.
Reference copying synchronizes execution; this does not establish performance, absence of timing-related bugs, generated-token extraction, or model accuracy.
Earlier eager tests also checked layer selection against the model's ordinary computation; the graph test checks preservation of the exported values.
See [RESULTS.md](RESULTS.md) for outcomes.
