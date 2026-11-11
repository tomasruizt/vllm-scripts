#!/usr/bin/env bash
# One variant on one GPU: start the server once, run every (repeat, concurrency) cell, then GSM8K.
# Usage: run.sh VARIANT   (inside a chg reservation, from the run directory)
# Host paths and the model come from env (defaults: B300, Qwen3.6). Variants: {main,pr,hier,pr65004,pr65103}-{k7,k8,mtp3}.
# AIME_N>0 adds AIME 2025 avg@AIME_N with the server's default sampling and thinking on (needs MAX_MODEL_LEN ~40k).
set -euo pipefail

CODE=${CODE:-$HOME/code}
MAIN=${MAIN:-$CODE/topk-cuda-main}
AIPERF=${AIPERF:-$HOME/aiperf-venv/bin/aiperf}
DATASET=${DATASET:-/data/tomasruizt/benchmarks/deepseek-v4-dep4-av/dataset/throughput_1k.jsonl}
REPEATS=${REPEATS-1 2 3}  # empty: skip the AIPerf cells (e.g. GSM8K only)
CONCURRENCIES=${CONCURRENCIES:-1 8 32 64 128 256}
CONCURRENCIES=${CONCURRENCIES//,/ }  # commas allowed, to pass through quoted ssh commands
DURATION=${DURATION:-30}  # seconds at c <= 32, plus 10s per doubling above, since requests get longer
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
DSPARK=${DSPARK:-RedHatAI/Qwen3.6-35B-A3B-speculator.dspark}
TP=${TP:-1}
BENCH_YAML=${BENCH_YAML:-benchmark.yaml}
GSM8K_SAMPLING=${GSM8K_SAMPLING:-"--temperature 0.7 --top-p 0.8 --top-k 20"}
# PRESET=glm: GLM-5.3-Flash (TP=4, FP8 KV cache as on its model card) with its default top-p-only
# sampling (T=1.0, top-p 0.95).
if [[ ${PRESET:-} == glm ]]; then
    MODEL=zai-org/GLM-5.3-Flash DSPARK=RedHatAI/GLM-5.3-Flash-speculator.dspark-preview TP=4
    SERVER_ARGS=${GLM_SERVER_ARGS-"--kv-cache-dtype fp8"}
    BENCH_YAML=benchmark-glm.yaml GSM8K_SAMPLING="--temperature 1.0 --top-p 0.95"
    HUB_CACHE=${HUB_CACHE:-$HOME/.cache/huggingface/hub}
fi
# PRESET=qwen-default: Qwen3.6 with its generation_config sampling (T=1.0, top-k 20, top-p 0.95).
if [[ ${PRESET:-} == qwen-default ]]; then
    BENCH_YAML=benchmark-default.yaml GSM8K_SAMPLING="--temperature 1.0 --top-p 0.95 --top-k 20"
fi
GSM8K_QUESTIONS=${GSM8K_QUESTIONS:-1319}
AIME_N=${AIME_N:-0}
AIME_DATA=${AIME_DATA:-$HOME/datasets/aime25.jsonl}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
CUDAGRAPH_METRICS=${CUDAGRAPH_METRICS-1}  # set empty to drop --cudagraph-metrics
export HF_HUB_CACHE=${HUB_CACHE:-/data/tomasruizt/hub_cache}  # the login env sets its own HF_HUB_CACHE
SERVER_PID="" TELEMETRY_PID="" LOAD_PID="" NODE_PID="" APPS_PID=""

main() {
    local variant=$1
    VARIANT=$variant
    trap cleanup EXIT
    trap 'exit 143' TERM INT
    export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1
    VENV=${VENV_OVERRIDE:-$(venv_dir "$variant")}
    export PATH=$VENV/bin:$PATH:/usr/local/cuda/bin  # JIT builds need ninja and nvcc
    export PYTHONPATH=${SRC_OVERRIDE-$(source_dir "$variant")}
    local port=$((8800 + RANDOM % 100))
    mkdir -p "$variant"
    git -C "$PYTHONPATH" log -1 --format='%H %cs %s' > "$variant/commit.txt" 2>/dev/null || echo "${IMAGE:-unknown}" > "$variant/commit.txt"
    nvidia-smi --query-gpu=name,uuid --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" > "$variant/gpu.txt" || true
    date -u +%FT%TZ > "$variant/start.txt"
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv \
        -l 5 -i "${CUDA_VISIBLE_DEVICES:-0}" > "$variant/gpu-telemetry.csv" 2>&1 &
    TELEMETRY_PID=$!
    # Node-wide neighbors: every GPU and every compute process, to explain cells slowed by other jobs.
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv -l 30 > "$variant/node-gpus.csv" 2>&1 &
    NODE_PID=$!
    (while true; do date -u +%FT%TZ; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,name --format=csv,noheader; sleep 30; done) > "$variant/node-apps.txt" 2>&1 &
    APPS_PID=$!
    bash host_load.sh "$variant/host-load.txt" &  # read by summarize.py for the host_quiet gate
    LOAD_PID=$!
    start_server "$variant" "$port"
    date -u +%FT%TZ > "$variant/ready.txt"
    local repeat
    for repeat in $REPEATS; do run_repeat "$variant" "$port" "$repeat"; done
    run_gsm8k "$variant" "$port"
    run_aime "$variant" "$port"
}

source_dir() {
    case $1 in main-*) echo "$MAIN" ;; pr-*) echo "$CODE/topk-cuda-pr2" ;; hier-*) echo "$CODE/topk-triton-hier-e2e" ;; pr65004-*) echo "$CODE/topk-pr65004" ;; pr65103-*) echo "$CODE/topk-pr65103" ;; esac
}

# #65103 is built on its own base (2026-11-01), with that base's precompiled artifacts.
venv_dir() {
    case $1 in pr65103-*) echo "$CODE/topk-main-65103/.venv" ;; *) echo "$MAIN/.venv" ;; esac
}

# A suffix like pr-k8.2 names a repeated session. -nospec decodes without a drafter. -mtp3 uses the model's own MTP head with 3 drafts.
spec_config() {
    case $1 in
        *-mtp3*) echo '{"method":"mtp","num_speculative_tokens":3}' ;;
        *-mtp5*) echo '{"method":"mtp","num_speculative_tokens":5}' ;;  # GLM on ROCm (DSpark's KV layout clashes there)
        *-nospec*) echo "" ;;  # normal decoding (GSM8K of the sampler without spec decoding)
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
    sed "s|__PORT__|$port|g; s|__ARTIFACTS__|$PWD/$variant/artifacts/$cell|g; s|__CONCURRENCIES__|$concurrency|g; s|__DURATIONS__|$(duration "$concurrency")|g; s|__DATASET__|$DATASET|g" "$BENCH_YAML" > "$config"
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
        --max-concurrency 32 --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 \
        $GSM8K_SAMPLING --save-results "$PWD/$variant/gsm8k.json" > "$variant/gsm8k.log" 2>&1
}

run_aime() {
    local variant=$1 port=$2
    (( AIME_N > 0 )) || return 0
    printf 'AIME %s %s\n' "$variant" "$(date -u +%FT%TZ)"
    timeout 7200 python aime_eval.py --port "$port" --data "$AIME_DATA" --n "$AIME_N" \
        --max-tokens $((MAX_MODEL_LEN - 1024)) --out "$PWD/$variant/aime.json" > "$variant/aime.log" 2>&1
}

start_server() {
    local variant=$1 port=$2
    local spec
    spec=$(spec_config "$variant")
    local command=("$VENV/bin/python" -m vllm.entrypoints.openai.api_server
        --model "$MODEL" --served-model-name served --language-model-only --tensor-parallel-size "$TP"
        --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization 0.85
        --no-enable-prefix-caching --disable-uvicorn-access-log ${CUDAGRAPH_METRICS:+--cudagraph-metrics}
        ${SERVER_ARGS:-} ${spec:+--speculative-config "$spec"} --host 127.0.0.1 --port "$port")
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
    kill $TELEMETRY_PID $LOAD_PID $NODE_PID $APPS_PID 2>/dev/null || true
    [[ -z "$SERVER_PID" ]] || { kill -TERM -- "-$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
    printf '%s\n' "$result" > "$variant/exit-code.txt"
    exit "$result"
}

main "$@"
