#!/usr/bin/env bash
set -euo pipefail
export ROOT=${ROOT:-/data/tomasruizt/runs/hopper-speedbench-capture-20261020}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export PYTHONPATH=/data/tomasruizt/runtime/vllm-capture-grid-58b3298457
export HF_HUB_CACHE=/data/tomasruizt/hub_cache
export XDG_CACHE_HOME=/data/tomasruizt/cache
export TMPDIR=/data/tomasruizt/tmp
export VLLM_SERVER_DEV_MODE=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="/data/tomasruizt/capture-env/bin:$HOME/.venv/bin:$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
unset VLLM_USE_V2_MODEL_RUNNER
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
PIDS=()
for model in dense gdn; do
    for variant in default 4k; do
        chg run --gpus 1 --wait 30m --timeout 90m --note "Hopper SPEED-Bench AIPerf $model $variant" -- /data/tomasruizt/capture-env/bin/python run.py "$model" "$variant" > "$model-$variant-job.log" 2>&1 &
        PIDS+=("$!")
    done
done
trap 'kill -TERM "${PIDS[@]}" 2>/dev/null || true; exit 143' TERM INT
RESULT=0
for pid in "${PIDS[@]}"; do wait "$pid" || RESULT=1; done
exit "$RESULT"
