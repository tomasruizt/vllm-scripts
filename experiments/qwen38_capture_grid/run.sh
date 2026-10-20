#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$HOME/runs/qwen38-capture-grid-20261019}
SOURCE=${SOURCE:-$HOME/code/vllm-qwen38-capture-grid}
MODEL=${MODEL:-/data/tomasruizt/hub_cache/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540}
PORT=${PORT:-8748}
CONCURRENCIES=${CONCURRENCIES:-32,128}
DURATIONS=${DURATIONS:-30,30}
AIPERF=$HOME/aiperf-venv/bin/aiperf
SERVER_PID=""

main() {
    cd "$ROOT"
    if [[ ${1:-} == prepare ]]; then prepare; return; fi
    [[ ! -e exit-code.txt ]] || { echo "Run directory already completed, choose a new ROOT" >&2; exit 1; }
    trap cleanup EXIT
    trap 'exit 143' TERM INT
    export PATH="$HOME/.venv/bin:$HOME/.local/bin:$PATH"
    export PYTHONPATH="$SOURCE"
    export VLLM_USE_V2_MODEL_RUNNER=1
    "$HOME/.venv/bin/python" -c 'import vllm; print(vllm.__file__, vllm.__version__)' > vllm-import.txt
    nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > gpu-inventory.csv
    for variant in default 2k 4k; do run_variant "$variant"; done
}

prepare() {
    mkdir -p configs
    local variant
    for variant in default 2k 4k; do
        sed "s|__PORT__|$PORT|g; s|__MODEL__|$MODEL|g; s|__ARTIFACTS__|$ROOT/$variant/artifacts|g; s|__CONCURRENCIES__|$CONCURRENCIES|g; s|__DURATIONS__|$DURATIONS|g" benchmark.yaml > "configs/$variant.yaml"
        "$AIPERF" config validate "configs/$variant.yaml" > "configs/$variant-validation.txt" 2>&1
        "$AIPERF" config expand "configs/$variant.yaml" > "configs/$variant-expanded.txt" 2>&1
    done
}

run_variant() {
    local variant=$1
    mkdir -p "$variant"
    date -u +%FT%TZ > "$variant/start.txt"
    printf 'START %s %s\n' "$variant" "$(date -u +%FT%TZ)"
    start_server "$variant"
    date -u +%FT%TZ > "$variant/ready.txt"
    printf 'READY %s %s\n' "$variant" "$(date -u +%FT%TZ)"
    timeout 900 "$AIPERF" profile --config "configs/$variant.yaml" > "$variant/client.log" 2>&1
    stop_server
    date -u +%FT%TZ > "$variant/end.txt"
    printf 'DONE %s %s\n' "$variant" "$(date -u +%FT%TZ)"
}

start_server() {
    local variant=$1
    if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
        echo "Port $PORT already serves a model, aborting" >&2
        exit 1
    fi
    local command=("$HOME/.venv/bin/python" -m vllm.entrypoints.openai.api_server
        --model "$MODEL" --served-model-name qwen38 --language-model-only --dtype bfloat16
        --tensor-parallel-size 4 --enable-expert-parallel --gpu-memory-utilization 0.90
        --max-model-len 16384 --max-num-seqs 256 --max-num-batched-tokens 8192
        --no-enable-prefix-caching --disable-uvicorn-access-log --cudagraph-metrics --kv-cache-dtype fp8
        --attention-config.indexer_kv_dtype fp8 --engram-config '{"cpu_offload":true}'
        --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
        --host 127.0.0.1 --port "$PORT")
    if [[ "$variant" != default ]]; then
        command+=(--compilation-config "{\"cudagraph_capture_sizes\":[$(capture_sizes "$variant")]}")
    fi
    printf '%q ' "${command[@]}" > "$variant/server-command.txt"
    printf '\n' >> "$variant/server-command.txt"
    setsid "${command[@]}" > "$variant/server.log" 2>&1 &
    SERVER_PID=$!
    local deadline=$((SECONDS + 900))
    until curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 20 "$variant/server.log"; exit 1; }
        (( SECONDS < deadline )) || { echo "Server startup timed out" >&2; exit 1; }
        sleep 2
    done
}

capture_sizes() {
    local additions='1280 1536 1792 2048'
    [[ $1 != 4k ]] || additions+=' 2560 3072 3584 4096'
    printf '%s\n' "1 2 4 $(seq -s ' ' 8 8 248) $(seq -s ' ' 256 16 1024) $additions" | tr ' ' ','
}

stop_server() {
    [[ -n "$SERVER_PID" ]] || return 0
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

cleanup() {
    local result=$?
    trap - EXIT
    stop_server
    printf '%s\n' "$result" > "$ROOT/exit-code.txt"
    printf 'EXIT %s %s\n' "$result" "$(date -u +%FT%TZ)"
    exit "$result"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
