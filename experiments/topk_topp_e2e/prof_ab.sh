#!/usr/bin/env bash
# Torch-profile a few steps of one variant at concurrency C, to compare the GPU time of the
# top-k/top-p kernels per step between builds. Usage (inside a chg reservation, from the run dir):
#   prof_ab.sh VARIANT [C]   (VARIANT: main-k7 or hier-k7, Qwen3.6 + DSpark K=7, default sampling)
set -euo pipefail
variant=$1 concurrency=${2:-256}
CODE=$HOME/code
case $variant in main-*) src=$CODE/topk-cuda-main ;; hier-*) src=$CODE/topk-triton-hier-e2e ;; esac
export PATH=$CODE/topk-cuda-main/.venv/bin:$PATH:/usr/local/cuda/bin PYTHONPATH=$src
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_CACHE=/data/tomasruizt/hub_cache
port=$((8900 + RANDOM % 90)) out=$PWD/$variant-c$concurrency
mkdir -p "$out"
profiler="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$out/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_dump_cuda_time_total\":false}"
setsid python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.6-35B-A3B --served-model-name served \
    --language-model-only --max-model-len 8192 --max-num-seqs 256 --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching --disable-uvicorn-access-log --profiler-config "$profiler" \
    --speculative-config '{"method":"dspark","model":"RedHatAI/Qwen3.6-35B-A3B-speculator.dspark","num_speculative_tokens":7}' \
    --host 127.0.0.1 --port $port > "$out/server.log" 2>&1 &
server=$!
trap 'kill -TERM -- -$server 2>/dev/null || true; wait $server 2>/dev/null || true' EXIT
until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null; do
    kill -0 $server 2>/dev/null || { tail -n 20 "$out/server.log"; exit 1; }
    sleep 5
done
sed "s|__PORT__|$port|g; s|__ARTIFACTS__|$out/aiperf|g; s|__CONCURRENCIES__|$concurrency|g; s|__DURATIONS__|90|g; s|__DATASET__|/data/tomasruizt/benchmarks/deepseek-v4-dep4-av/dataset/throughput_1k.jsonl|g" \
    benchmark-default.yaml > "$out/aiperf.yaml"
timeout 300 $HOME/aiperf-venv/bin/aiperf profile --config "$out/aiperf.yaml" > "$out/aiperf.log" 2>&1 &
client=$!
sleep 45  # steady state: every request slot busy
curl -sf -X POST http://127.0.0.1:$port/start_profile
sleep 3
curl -sf -X POST http://127.0.0.1:$port/stop_profile
wait $client || true
echo PROFILE_DONE
