# Commands

server:

```shell
vllm serve Qwen/Qwen3-32B \
    --no-enable-prefix-caching \
    --speculative_config.method=draft_model \
    --speculative_config.model=Qwen/Qwen3-1.7B \
    --speculative_config.num_speculative_tokens=4 \
    --speculative_config.max_model_len=5000 \
    --max-model-len 5000 \
    --max-num-seqs 1 \
    --profiler_config.profiler=torch \
    --profiler_config.torch_profiler_dir=./vllm_profile \
    --disable-uvicorn-access-log | tee results/serve.log
```

server 32B:

```shell
vllm serve Qwen/Qwen3-32B \
    --no-enable-prefix-caching \
    --max-model-len 5000 \
    --max-num-seqs 1 \
    --profiler_config.profiler=torch \
    --profiler_config.torch_profiler_dir=./vllm_profile \
    --disable-uvicorn-access-log | tee results/serve-32B.log
```

server 1.7B:

```shell
vllm serve Qwen/Qwen3-1.7B \
    --no-enable-prefix-caching \
    --max-model-len 5000 \
    --max-num-seqs 1 \
    --profiler_config.profiler=torch \
    --profiler_config.torch_profiler_dir=./vllm_profile \
    --disable-uvicorn-access-log | tee results/serve-1.7B.log
```

benchmark:

```shell
vllm bench serve \
    --model Qwen/Qwen3-32B \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num-prompts 1 \
    --output-len 1000 \
    --max-concurrency 1 \
    --temperature 0.0 \
    --top-p 1.0 \
    --profile \
    --ready-check-timeout-sec 600 | tee results/profile-nosd-32B.log
```