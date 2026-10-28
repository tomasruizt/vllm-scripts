#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
base="${CONTROL_RUN_ROOT:-$HOME/runs/gdn-radix-k15-20261027-1813-tp2}"
test -f "$base/DONE"
test "$(head -n 1 "$base/exit-code.txt")" = 0
export TARGET_MODEL_DIR=/data/tomasruizt/hub_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export DRAFT_MODEL_DIR=/data/tomasruizt/hub_cache/models--RadixArk--Qwen3.8-27B-DSpark/snapshots/b9a5dbdf03bc999c6c73c426b19c2d9041cea393
export TENSOR_PARALLEL_SIZE=2
export CONTROL_MODES=""
export CONTROL_RUN_ROOT="$base"
export BASELINE_RUN_ROOT="$base"
export PROFILE_MODES="fixed adaptive"
export RUN_ROOT="$base/c1-study"
mkdir -p "$RUN_ROOT"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits --loop-ms=1000 > "$RUN_ROOT/gpu-clocks.csv" 2> "$RUN_ROOT/gpu-clocks.log" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
"$HOME/.venv/bin/python" "$here/c1_study.py"
