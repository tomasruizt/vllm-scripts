#!/usr/bin/env bash
# Probe the KV budget of Qwen3-8B + XPress b16 on one H100 to size the concurrency ladder.
set -euo pipefail
export HF_HUB_CACHE=${HF_CACHE:-/mnt/nfs-preprod-1/engine/hub_cache}
export XDG_CACHE_HOME=$HOME/cache TMPDIR=$HOME/tmp
export PYTHONPATH=$HOME/code/vllm-xpress-av
export PATH="$HOME/code/vllm-dsv41/.venv/bin:/usr/local/cuda-13.0/bin:$PATH"
cd "$HOME"
spec='{"method":"dflash","model":"UIUC-SSAIL/Qwen3-8B-XPress-b16","num_speculative_tokens":15,"enable_adaptive_verification":true}'
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=${CONTEXT_LEN:-2048}
setsid "$HOME/code/vllm-dsv41/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --served-model-name qwen --max-model-len 4096 \
    --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --disable-uvicorn-access-log --host 127.0.0.1 --port "${PORT:-8670}" \
    --speculative-config "$spec" > "$HOME/runs/kv-probe.log" 2>&1 &
pid=$!
trap 'kill -TERM -- -$pid 2>/dev/null || true' EXIT
for ((i = 0; i < 600; i++)); do
    kill -0 "$pid" 2>/dev/null || { tail -20 "$HOME/runs/kv-probe.log"; exit 1; }
    curl -sf --max-time 1 "http://127.0.0.1:${PORT:-8670}/health" >/dev/null && break
    sleep 1
done
grep -a -E "KV cache size|Maximum concurrency|memory profiling|init engine" "$HOME/runs/kv-probe.log" | tail -5
