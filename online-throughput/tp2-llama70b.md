serve SD methods:

```shell
mkdir -p tp-results

vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 2 \
  --max-num-seqs 256 \
  --max-model-len 5000 \
  --disable-uvicorn-access-log \
  --no-enable-prefix-caching | tee tp-results/nosd-serve.log

K=8
mkdir -p k$K/tp-results-draft-model
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 2 \
  --speculative_config.method draft_model \
  --speculative_config.model meta-llama/Llama-3.2-1B \
  --speculative_config.num_speculative_tokens $K \
  --speculative_config.max_model_len 5000 \
  --max-num-seqs 256 \
  --max-model-len 5000 \
  --disable-uvicorn-access-log \
  --no-enable-prefix-caching | tee k$K/tp-results-draft-model/draft-model-serve.log

K=7
mkdir -p k$K/tp-results-eagle3
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 2 \
  --speculative_config.method eagle3 \
  --speculative_config.model yuhuili/EAGLE3-LLaMA3.3-Instruct-70B \
  --speculative_config.num_speculative_tokens $K \
  --max-num-seqs 256 \
  --max-model-len 5000 \
  --disable-uvicorn-access-log \
  --no-enable-prefix-caching | tee k$K/tp-results-eagle3/eagle3-serve.log
```

Note: I had to remove `--speculative_config.max_model_len` for method=eagle3 because it was causing the server to crash with a `ValidationError`:

```shell
(APIServer pid=31288) pydantic_core._pydantic_core.ValidationError: 1 validation error for SpeculativeConfig
(APIServer pid=31288)   Value error, speculative_max_model_len=5000 cannot be larger than draft_max_model_len=2048 [type=value_error, input_value=ArgsKwargs((), {'method':..., _api_process_rank=0)}), input_type=ArgsKwargs]
```

benchmark:

```shell
K=8
METHOD=eagle3
for MAX_CONCURRENCY in 1 2 4 8 16 32 64; do

  NUM_PROMPTS=$(( MAX_CONCURRENCY * 10 ))
  if [ $NUM_PROMPTS -lt 50 ]; then
    NUM_PROMPTS=50
  fi

  echo "Starting benchmark with MAX_CONCURRENCY = $MAX_CONCURRENCY and NUM_PROMPTS = $NUM_PROMPTS..."
  vllm bench serve \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num-prompts $NUM_PROMPTS \
    --max-concurrency $MAX_CONCURRENCY \
    --request-rate $MAX_CONCURRENCY \
    --temperature 0.0 \
    --top-p 1.0 \
    --ready-check-timeout-sec 600 | tee k$K/tp-results-$METHOD/bench-$METHOD-c$MAX_CONCURRENCY.log
done
```
