#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model=${1:?Specify 4B, 27B or 35B-A3B}
output=${2:?Specify output directory}
port=${3:?Specify port}
proposals=${4:-15}
variants=${5:-"vllm_baseline vllm_dflash sglang_baseline sglang_dflash pr2_dflash"}
code_dir=/home/tomasruizt/code
pids=()
mkdir -p "$output"
for variant in $variants; do
    engine=${variant%%_*}
    mode=${variant#*_}
    source_args=()
    case "$engine" in
        vllm) engine_python="$code_dir/vllm-main/.venv/bin/python" ;;
        sglang) engine_python="$code_dir/sglang-bench/.venv/bin/python" ;;
        pr2)
            engine=vllm
            engine_python="$HOME/.venv/bin/python"
            source_args=(--vllm-dir "$code_dir/vllm-pr2")
            ;;
    esac
    canhazgpu run --gpus 1 -- "$engine_python" "$script_dir/run.py" "$engine" "$mode" \
        "${source_args[@]}" --model-size "$model" --port "$port" \
        --num-speculative-tokens "$proposals" \
        --concurrencies 1 2 4 8 16 32 --gpu-memory-utilization 0.92 \
        --max-num-seqs 32 --sglang-max-running-requests 32 \
        --prefill-chunk-size 2048 \
        --draft-attention-backend TRITON_ATTN \
        --cudagraph-capture-sizes 1 2 4 8 16 32 64 128 256 512 \
        --output "$output/$model/$variant" > "$output/$model-$variant.log" 2>&1 &
    pids+=("$!")
    port=$((port + 1))
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
