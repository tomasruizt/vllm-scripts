#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/code/vllm/.venv/bin:$PATH"

python_bin="$HOME/code/vllm/.venv/bin/python"
mkdir results
"$python_bin" -m pip freeze > results/packages.txt
git -C "$HOME/code/vllm" rev-parse HEAD > results/vllm-commit.txt
git -C "$HOME/code/vllm" status --short > results/vllm-status.txt
sha256sum benchmark_topk_topp.py > results/benchmark-sha256.txt
nvidia-smi --query-gpu=index,name,uuid --format=csv > results/gpus.csv
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > results/gpu-ids.txt

if (( $# )); then
    "$python_bin" benchmark_topk_topp.py "$@" --output results/custom-r0.jsonl > results/custom-r0.log 2>&1
    touch results/DONE
    exit 0
fi

run_case() {
    local label="$1"
    local order="$2"
    shift 2
    local command=("$python_bin" benchmark_topk_topp.py --batch-sizes 4 9 32 192 256 280 --vocab-sizes 248320
        --warmup-iters 10 --benchmark-iters 200 --check --output "results/$label.jsonl")
    read -r -a implementations <<< "$order"
    command+=(--implementations "${implementations[@]}" "$@")
    printf '%q ' "${command[@]}" > "results/$label-command.txt"
    printf '\n' >> "results/$label-command.txt"
    "${command[@]}" > "results/$label.log" 2>&1
}

run_case dense-r0 'triton flashinfer-mask pytorch' --top-k 20 --top-p 0.8
run_case dense-r1 'flashinfer-mask pytorch triton' --top-k 20 --top-p 0.8
run_case dense-r2 'pytorch triton flashinfer-mask' --top-k 20 --top-p 0.8
run_case sparse-r0 'triton flashinfer-mask pytorch' --top-k 20 --top-p 0.8 --num-finite-logits 13
run_case defaults-r0 'triton flashinfer-mask pytorch' --batch-sizes 4 9 --benchmark-iters 20
touch results/DONE
