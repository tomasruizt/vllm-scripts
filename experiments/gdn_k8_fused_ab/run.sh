#!/usr/bin/env bash
# One variant on one GPU: start the server once, run every (repeat, concurrency) cell, then GSM8K.
# Usage: run.sh VARIANT   (inside a chg reservation, from the run directory)
# Host paths come from env (defaults: B300). Variants: {main,pr,splitv,splitv2}-{k7,k8,mtp3}.
set -euo pipefail

CODE=${CODE:-$HOME/code}
MAIN=${MAIN:-$CODE/gdn-k8-verify-b300-main}
AIPERF=${AIPERF:-$HOME/aiperf-venv/bin/aiperf}
DATASET=${DATASET:-/data/tomasruizt/benchmarks/deepseek-v4-dep4-av/dataset/throughput_1k.jsonl}
REPEATS=${REPEATS:-1 2 3}
CONCURRENCIES=${CONCURRENCIES:-1 2 4 8 16 32}
CONCURRENCIES=${CONCURRENCIES//,/ }  # commas allowed, to pass through quoted ssh commands
DURATION=${DURATION:-30}  # seconds at c <= 32, plus 10s per doubling above, since requests get longer
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
DSPARK=RedHatAI/Qwen3.6-35B-A3B-speculator.dspark
GSM8K_QUESTIONS=${GSM8K_QUESTIONS:-1319}
export HF_HUB_CACHE=${HUB_CACHE:-/data/tomasruizt/hub_cache}  # the login env sets its own HF_HUB_CACHE
SERVER_PID="" TELEMETRY_PID="" LOAD_PID=""

main() {
    local variant=$1
    VARIANT=$variant
    trap cleanup EXIT
    trap 'exit 143' TERM INT
    export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1
    export PATH=$MAIN/.venv/bin:$PATH:/usr/local/cuda/bin  # JIT builds need ninja and nvcc
    export PYTHONPATH=$(source_dir "$variant")
    local port=$((8800 + RANDOM % 100))
    mkdir -p "$variant"
    git -C "$PYTHONPATH" log -1 --format='%H %cs %s' > "$variant/commit.txt"
    nvidia-smi --query-gpu=name,uuid --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" > "$variant/gpu.txt" || true
    date -u +%FT%TZ > "$variant/start.txt"
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv \
        -l 5 -i "${CUDA_VISIBLE_DEVICES:-0}" > "$variant/gpu-telemetry.csv" 2>&1 &
    TELEMETRY_PID=$!
    (while sleep 5; do echo "$(date -u +%T) $(cat /proc/loadavg)"; done) > "$variant/host-load.txt" 2>&1 &
    LOAD_PID=$!
    start_server "$variant" "$port"
    date -u +%FT%TZ > "$variant/ready.txt"
    local repeat
    for repeat in $REPEATS; do run_repeat "$variant" "$port" "$repeat"; done
    run_gsm8k "$variant" "$port"
}

source_dir() {
    case $1 in main-*) echo "$MAIN" ;; pr-*) echo "$CODE/gdn-k8-pr-on-main" ;; *) echo "$CODE/gdn-k8-${1%-*}" ;; esac
}

# A suffix like pr-k8.2 names a repeated session. -mtp3 uses the model's own MTP head with 3 drafts.
spec_config() {
    case $1 in
        *-mtp3*) echo '{"method":"mtp","num_speculative_tokens":3}' ;;
        *-k7*) echo "{\"method\":\"dspark\",\"model\":\"$DSPARK\",\"num_speculative_tokens\":7}" ;;
        *) echo "{\"method\":\"dspark\",\"model\":\"$DSPARK\",\"num_speculative_tokens\":8}" ;;
    esac
}

run_repeat() {
    local variant=$1 port=$2 repeat=$3 order=$CONCURRENCIES concurrency
    [[ $repeat != 2 ]] || order=$(echo "$CONCURRENCIES" | tr ' ' '\n' | tac | tr '\n' ' ')
    for concurrency in $order; do run_cell "$variant" "$port" "r$repeat-c$concurrency" "$concurrency"; done
}

run_cell() {
    local variant=$1 port=$2 cell=$3 concurrency=$4
    local config="$variant/configs/$cell.yaml"
    mkdir -p "$(dirname "$config")"
    sed "s|__PORT__|$port|g; s|__ARTIFACTS__|$PWD/$variant/artifacts/$cell|g; s|__CONCURRENCIES__|$concurrency|g; s|__DURATIONS__|$(duration "$concurrency")|g; s|__DATASET__|$DATASET|g" benchmark.yaml > "$config"
    printf 'CELL %s %s %s\n' "$variant" "$cell" "$(date -u +%FT%TZ)"
    timeout 300 "$AIPERF" profile --config "$config" > "$variant/$cell-client.log" 2>&1
}

duration() {
    local concurrency=$1 seconds=$DURATION level=32
    while (( concurrency > level )); do seconds=$((seconds + 10)); level=$((level * 2)); done
    echo "$seconds"
}

run_gsm8k() {
    local variant=$1 port=$2
    (( GSM8K_QUESTIONS > 0 )) || return 0
    printf 'GSM8K %s %s\n' "$variant" "$(date -u +%FT%TZ)"
    timeout 1200 python "$MAIN/tests/evals/gsm8k/gsm8k_eval.py" --port "$port" --num-questions "$GSM8K_QUESTIONS" \
        --max-concurrency 32 --save-results "$PWD/$variant/gsm8k.json" > "$variant/gsm8k.log" 2>&1
}

start_server() {
    local variant=$1 port=$2
    local spec
    spec=$(spec_config "$variant")
    local command=("$MAIN/.venv/bin/python" -m vllm.entrypoints.openai.api_server
        --model "$MODEL" --served-model-name qwen --language-model-only
        --max-model-len 8192 --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization 0.85
        --no-enable-prefix-caching --disable-uvicorn-access-log --cudagraph-metrics
        --speculative-config "$spec" --host 127.0.0.1 --port "$port")
    printf '%q ' "${command[@]}" > "$variant/server-command.txt"
    setsid "${command[@]}" > "$variant/server.log" 2>&1 &
    SERVER_PID=$!
    local deadline=$((SECONDS + 1800))
    until curl --silent --fail --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 20 "$variant/server.log"; exit 1; }
        (( SECONDS < deadline )) || { echo "Startup timed out" >&2; exit 1; }
        sleep 5
    done
}

cleanup() {
    local result=$? variant=$VARIANT
    trap - EXIT
    kill $TELEMETRY_PID $LOAD_PID 2>/dev/null || true
    [[ -z "$SERVER_PID" ]] || { kill -TERM -- "-$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
    printf '%s\n' "$result" > "$variant/exit-code.txt"
    exit "$result"
}

main "$@"
