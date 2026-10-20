#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-$HOME/runs/qwen38-recipe-restarts-20261020}
cd "$BASE"
WORKER=${1:?Specify worker 0 or 1}
trap 'printf "%s\n" "$?" > "worker-$WORKER-exit-code.txt"' EXIT
trap 'exit 143' TERM INT
export REPEATS=1
export PORT=$((8788 + WORKER))
export SOURCE=$HOME/code/vllm-qwen38-capture-grid

run_startup() {
    local startup=$1
    local variant=$2
    export ROOT=$BASE/qwen38-recipe-startup-$startup-$variant
    export VARIANTS=$variant
    export CELL_ORDER='256 128 64 32 8 1'
    if [[ $startup == 3 ]]; then
        export CELL_ORDER='1 8 32 64 128 256'
    fi
    mkdir -p "$ROOT"
    cp recipe-run.sh run.sh benchmark.yaml runtime-commit.txt "$ROOT/"
    printf '%s\n' "$startup" > "$ROOT/startup-id.txt"
    ROOT="$ROOT" bash "$ROOT/recipe-run.sh" prepare
    printf 'STARTUP %s %s worker=%s %s\n' "$startup" "$variant" "$WORKER" "$(date -u +%FT%TZ)"
    ROOT="$ROOT" bash "$ROOT/recipe-run.sh"
}

if [[ $WORKER == 0 ]]; then
    run_startup 2 default
    run_startup 2 4k
    run_startup 3 2k
else
    [[ $WORKER == 1 ]]
    run_startup 2 2k
    run_startup 3 default
    run_startup 3 4k
fi
