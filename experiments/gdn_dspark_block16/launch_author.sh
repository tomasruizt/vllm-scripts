#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
export RUN_ROOT="${RUN_ROOT:-$HOME/runs/gdn-author-555978f8-20261027-2210-tp2}"
export SOURCE_DIR="$HOME/code/vllm-pr60229-author-555978f8"
export PATH="$HOME/.venv/bin:$HOME/.local/bin:$PATH"
export TARGET_MODEL_DIR=/data/tomasruizt/hub_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export DRAFT_MODEL_DIR=/data/tomasruizt/hub_cache/models--RadixArk--Qwen3.8-27B-DSpark/snapshots/b9a5dbdf03bc999c6c73c426b19c2d9041cea393
export TENSOR_PARALLEL_SIZE=2

main() {
    mkdir -p "$RUN_ROOT"
    if [ "${1:-}" != --reserved ]; then
        exec chg run --gpus 8 --wait 10m --timeout 90m --note gdn-author-555978f8 -- bash "$0" --reserved
    fi
    run_reserved
}

run_reserved() {
    local base="$RUN_ROOT" failed=0
    local -a gpu_ids pids
    IFS=, read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    test "${#gpu_ids[@]}" = 8
    test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = 555978f8cf2cd21b22adba5d09d5f6e9df357714
    test -z "$(git -C "$SOURCE_DIR" status --porcelain)"
    git -C "$SOURCE_DIR" rev-parse HEAD > "$base/source-head.txt"
    git -C "$SOURCE_DIR" diff > "$base/source.patch"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.total,driver_version --format=csv > "$base/hardware.csv"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits --loop-ms=1000 > "$base/gpu-clocks.csv" 2> "$base/gpu-clocks.log" &
    local monitor_pid=$!
    trap "kill $monitor_pid 2>/dev/null || true" EXIT
    arm fixed "${gpu_ids[0]},${gpu_ids[1]}" 8796 "$base" & pids+=("$!")
    arm baseline "${gpu_ids[2]},${gpu_ids[3]}" 8797 "$base" & pids+=("$!")
    arm adaptive "${gpu_ids[4]},${gpu_ids[5]}" 8798 "$base" & pids+=("$!")
    diagnostics "${gpu_ids[6]},${gpu_ids[7]}" "$base" & pids+=("$!")
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    printf '%s\n' "$failed" > "$base/exit-code.txt"
    if [ "$failed" = 0 ]; then touch "$base/DONE"; fi
    return "$failed"
}

arm() {
    local mode="$1" devices="$2" port="$3" base="$4"
    export CUDA_VISIBLE_DEVICES="$devices" BENCH_PORT="$port"
    export BENCH_TMPDIR="/tmp/gdn-author-$port"
    export RUN_ROOT="$base/modes/$mode" VALIDATION_MODES="$mode" RUN_METADATA_TESTS=0
    mkdir -p "$RUN_ROOT"
    printf '%s\n' "$devices" > "$RUN_ROOT/gpu-ids.txt"
    "$HOME/.venv/bin/python" "$here/run.py" > "$RUN_ROOT/runner.log" 2>&1
    mv "$RUN_ROOT/$mode" "$base/$mode"
}

diagnostics() {
    local devices="$1" base="$2"
    export CUDA_VISIBLE_DEVICES="$devices" BENCH_PORT=8799
    export BENCH_TMPDIR=/tmp/gdn-author-8799
    export RUN_ROOT="$base/modes/diagnostic" VALIDATION_MODES=diagnostic RUN_METADATA_TESTS=1
    mkdir -p "$RUN_ROOT"
    printf '%s\n' "$devices" > "$RUN_ROOT/gpu-ids.txt"
    "$HOME/.venv/bin/python" "$here/run.py" > "$RUN_ROOT/runner.log" 2>&1
    mv "$RUN_ROOT/diagnostic" "$base/diagnostic"
    cp "$RUN_ROOT/metadata-tests.log" "$base/metadata-tests.log"
    export RUN_ROOT="$base/c1-study"
    mkdir -p "$RUN_ROOT"
    cd "$here"
    "$HOME/.venv/bin/python" - > "$RUN_ROOT/runner.log" 2>&1 <<'PY'
import os
import run as runner
from c1_study import profile

runner.configure_environment()
os.environ["BENCH_C1_ONLY"] = "1"
os.environ["PROFILE_C1"] = "1"
exit_code = 1
try:
    for mode in ("fixed", "adaptive"):
        profile(mode)
    (runner.ROOT / "DONE").touch()
    exit_code = 0
finally:
    runner.stop_server()
    (runner.ROOT / "exit-code.txt").write_text(str(exit_code))
PY
}

main "$@"
