#!/usr/bin/env bash
# Prefill + first-token latency of DSpark K=5 + adaptive verification, main vs branch, with `vllm bench latency`.
# Arms alternate (main, branch, main, branch) so drift over time hits both equally.
# An 8300-token prompt with an 8192-token step budget is prefilled in two chunks, like the server runs.
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export HF_HUB_CACHE=${HF_HUB_CACHE:?}
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/data/tomasruizt/cache
export TMPDIR=/data/tomasruizt/tmp
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$HOME/code/vllm-dsv41/.venv/bin:$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
unset VLLM_USE_V2_MODEL_RUNNER
TP=${TP:-4}
MODEL=$HF_HUB_CACHE/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
declare -A ARM_PYTHONPATH=([main]="" [branch]="$HOME/code/vllm-av-lazy-sync")

run_arm() {  # arm round
    local out="$1-round$2"
    mkdir "$out"
    cat /proc/loadavg > "$out/loadavg-before.txt"
    PYTHONPATH=${ARM_PYTHONPATH[$1]} chg run --gpus "$TP" --wait 24h --note "DSV4.1 AV prefill latency: $out" -- \
        vllm bench latency --model "$MODEL" --tensor-parallel-size "$TP" --enable-expert-parallel \
        --tokenizer-mode deepseek_v41 --max-model-len 16384 --max-num-batched-tokens 8192 --max-num-seqs 64 \
        --gpu-memory-utilization 0.88 --block-size 128 --safetensors-load-strategy prefetch --enable-ep-weight-filter \
        --compilation-config '{"max_cudagraph_capture_size":4096}' --engram-config '{"cpu_offload":true}' \
        --no-enable-prefix-caching --speculative-config "$SPEC" \
        --input-len 8300 --output-len 1 --batch-size 1 --num-iters-warmup 5 --num-iters 30 \
        --output-json "$out/latency.json" > "$out/bench.log" 2>&1
    cat /proc/loadavg > "$out/loadavg-after.txt"
}

for round in 1 2; do
    for arm in main branch; do run_arm "$arm" "$round"; done
done
