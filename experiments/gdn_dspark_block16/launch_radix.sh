#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
export RUN_ROOT="${RUN_ROOT:-$HOME/runs/gdn-radix-k15-20261027-1813-tp2}"
export PATH="$HOME/.venv/bin:$HOME/.local/bin:$PATH"

main() {
    mkdir -p "$RUN_ROOT"
    if [ "${1:-}" = --reserved ]; then
        run_reserved
        return
    fi
    wait_for_models
    chg run --gpu-ids 5,6 --timeout 2h --note gdn-radix-k15 -- bash "$0" --reserved
}

run_reserved() {
    export TARGET_MODEL_DIR=/data/tomasruizt/hub_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
    export DRAFT_MODEL_DIR=/data/tomasruizt/hub_cache/models--RadixArk--Qwen3.8-27B-DSpark/snapshots/b9a5dbdf03bc999c6c73c426b19c2d9041cea393
    export TENSOR_PARALLEL_SIZE=2
    export VALIDATION_MODES="fixed baseline adaptive diagnostic"
    source_dir="$HOME/code/vllm-dspark-block16-20261026-1745"
    git -C "$source_dir" rev-parse HEAD > "$RUN_ROOT/source-head.txt"
    git -C "$source_dir" diff > "$RUN_ROOT/source.patch"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits --loop-ms=1000 > "$RUN_ROOT/gpu-clocks.csv" 2> "$RUN_ROOT/gpu-clocks.log" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    "$HOME/.venv/bin/python" "$here/run.py"
}

wait_for_models() {
    deadline=$((SECONDS + 1800))
    until [ -f "$RUN_ROOT/MODELS_READY" ]; do
        if [ -f "$RUN_ROOT/prepare-exit.txt" ] && [ "$(head -n 1 "$RUN_ROOT/prepare-exit.txt")" != 0 ]; then
            echo "Model preparation failed, see prepare.log" >&2
            return 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "Model preparation exceeded 30 minutes" >&2
            return 1
        fi
        sleep 5
    done
}

main "$@"
