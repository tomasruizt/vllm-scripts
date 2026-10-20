#!/usr/bin/env bash
# Frozen-profile controls, with fresh startup AV, calibrated AV and fixed k=5.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export ROOT="${ROOT:-$HOME/runs/qwen-fixed-drafter-controls-20261019}"
export VLLM_CHECKOUT="$HOME/code/vllm-fixed-drafter-controls-60229"
export VLLM_DIAG_CHECKOUT="$HOME/code/vllm-cost-fixed-drafter-60229"
export VLLM_AV_COST_CURVES="$ROOT/cost-curves.json"
export CELLS="1:24 8:96 128:512 1:24"
export PORT_A=8682
export PORT_B=8683
export PLAN_A="1:oncg 1:offk5cg 2:oncostcg 1:oncgdiag 1:oncostcgdiag"
export PLAN_B="1:oncostcg 3:oncg 3:offk5cg 3:oncostcg"

main() {
    mkdir -p "$ROOT"
    cp "$here/run-steady.sh" "$here/steady-speedbench.yaml" "$ROOT/"
    cp "$here/run-fixed-drafter-controls.sh" "$ROOT/"
    echo "$CELLS" > "$ROOT/cells.txt"
    if [[ "${1:-}" == --prelude ]]; then
        launch 7 "$PORT_B" "2:oncg 2:offk5cg"
        return
    fi
    test -s "$VLLM_AV_COST_CURVES"
    sha256sum "$VLLM_AV_COST_CURVES" > "$ROOT/curve-sha256.txt"
    launch 6 "$PORT_A" "$PLAN_A" > "$ROOT/worker-gpu6.log" 2>&1 &
    local worker_a=$!
    timeout 1200 bash -c 'until grep -qx 0 "$ROOT/worker-gpu7-exit-code.txt"; do sleep 2; done'
    launch 7 "$PORT_B" "$PLAN_B" > "$ROOT/worker-gpu7.log" 2>&1 &
    local worker_b=$!
    wait "$worker_a"
    wait "$worker_b"
}

launch() {
    canhazgpu run --gpu-ids "$1" --timeout 3h --note "rh-2bm fixed-drafter runtime controls" -- \
        bash "$ROOT/run-steady.sh" --worker "$@"
}

main "$@"
