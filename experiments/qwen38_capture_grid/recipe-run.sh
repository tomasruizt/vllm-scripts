#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$HOME/runs/qwen38-recipe-grid-20261019}
PORT=${PORT:-8768}
MODEL=/data/tomasruizt/hub_cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13
source "$(dirname "${BASH_SOURCE[0]}")/run.sh"

main() {
    cd "$ROOT"
    export PATH="$HOME/.venv/bin:$HOME/.local/bin:$PATH"
    export PYTHONPATH="$SOURCE"
    export VLLM_SERVER_DEV_MODE=1
    unset VLLM_USE_V2_MODEL_RUNNER
    if [[ ${1:-} == prepare ]]; then prepare_recipe; return; fi
    [[ ! -e exit-code.txt ]]
    trap cleanup EXIT
    trap 'exit 143' TERM INT
    [[ -s runtime-commit.txt ]]
    nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > gpu-inventory.csv
    for variant in ${VARIANTS:-default 2k 4k}; do recipe_variant "$variant"; done
}

prepare_recipe() {
    local variant repeat concurrency duration output
    for variant in ${VARIANTS:-default 2k 4k}; do
        for repeat in ${REPEATS:-1 2 3}; do
            for concurrency in 1 8 32 64 128 256; do
                duration=30
                [[ $concurrency != 256 ]] || duration=60
                output="configs/$variant/r$repeat-c$concurrency.yaml"
                mkdir -p "$(dirname "$output")"
                sed "s|__PORT__|$PORT|g; s|__MODEL__|$MODEL|g; s|__ARTIFACTS__|$ROOT/$variant/artifacts/r$repeat-c$concurrency|g; s|__CONCURRENCIES__|$concurrency|g; s|__DURATIONS__|$duration|g" benchmark.yaml > "$output"
                "$AIPERF" config validate "$output" > "${output%.yaml}-validation.txt" 2>&1
            done
        done
    done
}

recipe_variant() {
    local variant=$1
    local repeat concurrency order
    mkdir -p "$variant"
    date -u +%FT%TZ > "$variant/start.txt"
    printf 'START %s %s\n' "$variant" "$(date -u +%FT%TZ)"
    start_recipe_server "$variant"
    date -u +%FT%TZ > "$variant/ready.txt"
    curl --silent --fail "http://127.0.0.1:$PORT/server_info?config_format=json" > "$variant/server-info.json"
    printf 'READY %s %s\n' "$variant" "$(date -u +%FT%TZ)"
    for repeat in ${REPEATS:-1 2 3}; do
        order=${CELL_ORDER:-256 128 64 32 8 1}
        [[ $repeat != 2 ]] || order='1 8 32 64 128 256'
        for concurrency in $order; do recipe_cell "$variant" "$repeat" "$concurrency"; done
    done
    stop_server
    date -u +%FT%TZ > "$variant/end.txt"
    printf 'DONE %s %s\n' "$variant" "$(date -u +%FT%TZ)"
}

recipe_cell() {
    local variant=$1
    local repeat=$2
    local concurrency=$3
    local cell="r$repeat-c$concurrency"
    curl --silent --show-error --fail --request POST "http://127.0.0.1:$PORT/reset_prefix_cache" > "$variant/$cell-cache-reset.txt"
    "$HOME/.venv/bin/python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["success"]' "$variant/$cell-cache-reset.txt"
    printf 'CELL %s %s %s\n' "$variant" "$cell" "$(date -u +%FT%TZ)"
    timeout 240 "$AIPERF" profile --config "configs/$variant/$cell.yaml" > "$variant/$cell-client.log" 2>&1
    printf 'CELL_DONE %s %s %s\n' "$variant" "$cell" "$(date -u +%FT%TZ)"
}

start_recipe_server() {
    local variant=$1
    if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
        echo "Port already in use" >&2
        exit 1
    fi
    local command=("$HOME/.venv/bin/python" -m vllm.entrypoints.openai.api_server
        --model "$MODEL" --served-model-name qwen38
        --tensor-parallel-size 4 --enable-expert-parallel --gpu-memory-utilization 0.90
        --max-num-seqs 256 --enable-prefix-caching --kv-cache-dtype fp8
        --attention-config.indexer_kv_dtype fp8
        --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
        --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
        --disable-uvicorn-access-log --cudagraph-metrics --host 127.0.0.1 --port "$PORT")
    if [[ $variant != default ]]; then
        command+=(--compilation-config "{\"cudagraph_capture_sizes\":[$(capture_sizes "$variant")]}")
    fi
    printf '%q ' "${command[@]}" > "$variant/server-command.txt"
    printf '\n' >> "$variant/server-command.txt"
    setsid "${command[@]}" > "$variant/server.log" 2>&1 &
    SERVER_PID=$!
    local deadline=$((SECONDS + 1200))
    until curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 20 "$variant/server.log"; exit 1; }
        (( SECONDS < deadline )) || { echo "Startup timed out" >&2; exit 1; }
        sleep 2
    done
}

main "$@"
