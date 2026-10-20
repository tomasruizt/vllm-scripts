#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$HOME/runs/glm53-flash-capture-pilot-20261019}
SOURCE=${SOURCE:-$HOME/code/vllm-qwen38-capture-grid}
MODEL=/data/engine/hub_cache/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
PORT=${PORT:-8778}
source "${HELPERS:-$(dirname "${BASH_SOURCE[0]}")/../qwen38_capture_grid/run.sh}"

main() {
    cd "$ROOT"
    export PATH="$HOME/.venv/bin:$HOME/.local/bin:$PATH"
    export PYTHONPATH="$SOURCE"
    export VLLM_SERVER_DEV_MODE=1
    unset VLLM_USE_V2_MODEL_RUNNER
    [[ ! -e exit-code.txt ]]
    trap cleanup EXIT
    trap 'exit 143' TERM INT
    prepare_configs
    nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > gpu-inventory.csv
    "$HOME/.venv/bin/python" -c 'import vllm; print(vllm.__version__, vllm.__file__)' > runtime.txt
    for variant in default 4k; do run_grid "$variant"; done
}

prepare_configs() {
    local variant repeat concurrency output
    for variant in default 4k; do
        for repeat in 1 2 3; do
            for concurrency in 64 128; do
                output="configs/$variant/r$repeat-c$concurrency.yaml"
                mkdir -p "$(dirname "$output")"
                sed "s|__PORT__|$PORT|g; s|__MODEL__|$MODEL|g; s|__ARTIFACTS__|$ROOT/$variant/artifacts/r$repeat-c$concurrency|g; s|__CONCURRENCIES__|$concurrency|g; s|__DURATIONS__|30|g" benchmark.yaml > "$output"
                "$AIPERF" config validate "$output" > "${output%.yaml}-validation.txt" 2>&1
            done
        done
    done
}

run_grid() {
    local variant=$1
    local repeat concurrency order cell
    mkdir -p "$variant"
    date -u +%FT%TZ > "$variant/start.txt"
    start_glm "$variant"
    date -u +%FT%TZ > "$variant/ready.txt"
    curl --silent --fail "http://127.0.0.1:$PORT/server_info?config_format=json" > "$variant/server-info.json"
    printf 'READY %s %s\n' "$variant" "$(date -u +%FT%TZ)"
    for repeat in 1 2 3; do
        order='128 64'
        [[ $repeat != 2 ]] || order='64 128'
        for concurrency in $order; do
            cell="r$repeat-c$concurrency"
            curl --silent --show-error --fail --request POST "http://127.0.0.1:$PORT/reset_prefix_cache" > "$variant/$cell-cache-reset.json"
            "$HOME/.venv/bin/python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["success"]' "$variant/$cell-cache-reset.json"
            printf 'CELL %s %s %s\n' "$variant" "$cell" "$(date -u +%FT%TZ)"
            timeout 240 "$AIPERF" profile --config "configs/$variant/$cell.yaml" > "$variant/$cell-client.log" 2>&1
            printf 'CELL_DONE %s %s %s\n' "$variant" "$cell" "$(date -u +%FT%TZ)"
        done
    done
    stop_server
    date -u +%FT%TZ > "$variant/end.txt"
    printf 'DONE %s %s\n' "$variant" "$(date -u +%FT%TZ)"
}

start_glm() {
    local variant=$1
    ! curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null
    local command=("$HOME/.venv/bin/python" -m vllm.entrypoints.openai.api_server
        --model "$MODEL" --served-model-name glm53-flash
        --tensor-parallel-size 4 --kv-cache-dtype fp8
        --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
        --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47
        --disable-uvicorn-access-log --cudagraph-metrics --host 127.0.0.1 --port "$PORT")
    if [[ $variant == 4k ]]; then
        "$HOME/.venv/bin/python" - <<'PY' > expanded-grid.json
import json
from pathlib import Path
info = json.loads(Path("default/server-info.json").read_text())
base = info["vllm_config"]["compilation_config"]["cudagraph_capture_sizes"]
assert base and max(base) <= 4096, base
sizes = sorted(set(base) | {1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096})
print(json.dumps({"cudagraph_capture_sizes": sizes}))
PY
        command+=(--compilation-config "$(<expanded-grid.json)")
    fi
    printf '%q ' "${command[@]}" > "$variant/server-command.txt"
    printf '\n' >> "$variant/server-command.txt"
    setsid "${command[@]}" > "$variant/server.log" 2>&1 &
    SERVER_PID=$!
    local deadline=$((SECONDS + 1200))
    until curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 20 "$variant/server.log"; exit 1; }
        (( SECONDS < deadline )) || { echo 'Startup timed out' >&2; exit 1; }
        sleep 2
    done
}

main "$@"
