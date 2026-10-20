#!/usr/bin/env bash
# One reserved GPU and one server per independent cost-data fold.
set -euo pipefail
source "$(dirname "$0")/run-steady.sh"

oracle_main() {
    local gpu=$1
    local port=$2
    local dir="$root/onoraclcg/repeat-1"
    mkdir -p "$dir" "$root/commands"
    rm -f "$root/worker-gpu$gpu-exit-code.txt"
    cd "$root"
    export PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH"
    export PYTHONPATH="$VLLM_CHECKOUT"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
    export AV_COST_CONTROL="$root/control.json"
    export AV_COST_TRACE="$root/cost-steps.jsonl"
    export AV_COST_SEED="${AV_COST_SEED:-42}"
    printf '{"mode":"native","phase":"startup"}\n' > "$AV_COST_CONTROL"
    git -C "$HOME/code/vllm-validate-60229" diff --exit-code HEAD
    git -C "$HOME/code/vllm-validate-60229" rev-parse HEAD > "$root/commit.txt"
    echo "$gpu" > "$dir/gpu.txt"
    trap "echo \$? > '$root/worker-gpu$gpu-exit-code.txt'; stop_server" EXIT
    trap 'exit 143' TERM INT
    start_server "$port" onoraclcg "$dir"
    "$HOME/.venv/bin/python" "$here/cost_oracle_queue.py" "$root" "$port" "$here/steady-speedbench.yaml"
}

oracle_main "$@"
