#!/usr/bin/env bash
set -euo pipefail

analysis_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
experiment_dir=$(cd -- "$analysis_dir/.." && pwd)
source "$HOME/.venv/bin/activate"
source "$experiment_dir/../../rh-setup/.bashrc-d/set-cuda-visible.bash"
if [[ -n "${ADAPTIVE_K_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$ADAPTIVE_K_CUDA_VISIBLE_DEVICES"
else
    set-cuda-visible
fi

export VLLM_USE_V2_MODEL_RUNNER=1

port=${ADAPTIVE_K_PORT:-8000}
experiment_name=${ADAPTIVE_K_EXPERIMENT_NAME:-qwen3_32b_fp8_tp2_complete}
num_runs=${ADAPTIVE_K_NUM_RUNS:-3}

serve_cmd="vllm serve Qwen/Qwen3-32B-FP8 --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --disable-uvicorn-access-log --port $port"
bench_cmd="vllm bench serve --backend openai-chat --base-url http://127.0.0.1:$port --endpoint /v1/chat/completions --model Qwen/Qwen3-32B-FP8 --tokenizer Qwen/Qwen3-32B-FP8 --dataset-name speed_bench --dataset-path $experiment_dir/data --speed-bench-dataset-subset qualitative --speed-bench-output-len 2048 --request-rate inf --skip-chat-template --temperature 1.0 --top-p 0.95 --seed 0 --save-detailed"

vllm bench sweep serve \
    --serve-cmd "$serve_cmd" \
    --bench-cmd "$bench_cmd" \
    --serve-params "$analysis_dir/serve_params.json" \
    --bench-params "$analysis_dir/bench_params.json" \
    --server-ready-timeout 900 \
    --show-stdout \
    --num-runs "$num_runs" \
    --output-dir "$experiment_dir/results" \
    --experiment-name "$experiment_name" \
    "$@"
