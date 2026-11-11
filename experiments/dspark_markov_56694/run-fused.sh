#!/usr/bin/env bash
# Single-GPU validation of the fused PR #56694 revision (head 0e5254884c).
# One server startup per mode per round, rotated mode order. Each cell sends a
# fixed request count so every arm processes the same prompt prefix.
# Artifacts land in <mode>_r<round>/<workload>/c<concurrency>/artifacts.
set -euo pipefail
root=${ROOT:?}
py=${PY:-$root/venv/bin/python}
client=$root/client-venv/bin/aiperf
port=${PORT:-8665}
rounds=${ROUNDS:-3}
modes=(${MODES:-dense union32 union32_unfused})
workloads=(${WORKLOADS:-8k 16k})
concurrencies=(${CONCURRENCIES:-1 4 8 32})
server_pid=""
monitor_pid=""
load_pid=""
export HF_HUB_CACHE=${HF_HUB_CACHE:?set HF_HUB_CACHE}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$root/venv/bin:$CUDA_HOME/bin:$PATH
export VLLM_USE_V2_MODEL_RUNNER=1
target=${TARGET:-$HF_HUB_CACHE/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}
draft=${DRAFT:-$HF_HUB_CACHE/models--AQ-MedAI--Qwen3-4B-DSpark/snapshots/49d6beedccdb7e2500382171461cb9868c2c5fbc}

main() {
    cd "$root"
    [[ ! -e exit-code.txt ]] || { echo "Use a fresh ROOT: $root already ran" >&2; exit 1; }
    trap 'finish $?' EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    nvidia-smi --query-gpu=index,uuid,name,compute_mode,power.limit --format=csv > gpu-info.csv
    run_tests
    for round in $(seq 1 "$rounds"); do
        for mode in $(rotate "$round"); do
            run_mode_round "$mode" "$round"
        done
    done
    probe_fusion
    touch DONE
}

# Rotate the mode order per round so no mode always runs first.
rotate() {
    local shift=$(( ($1 - 1) % ${#modes[@]} ))
    printf '%s\n' "${modes[@]:$shift}" "${modes[@]:0:$shift}"
}

source_for() {
    [[ $1 == union32_unfused ]] && echo pr-unfused || echo pr
}

run_mode_round() {
    local mode=$1 round=$2 dir
    dir=$root/${mode}_r${round}
    mkdir -p "$dir"
    start_server "$(source_for "$mode")" "$mode" "$dir"
    "$py" "$root/check_outputs.py" "$port" "$root/throughput_8k.jsonl" "$dir/outputs.json"
    for workload in "${workloads[@]}"; do run_workload "$dir" "$workload"; done
    stop_server
}

start_server() {
    local source=$1 mode=$2 dir=$3 spec
    spec='{"method":"dspark","model":"'"$draft"'","num_speculative_tokens":8}'
    [[ $mode != dense ]] && spec=${spec%\}}',"dspark_draft_topk":32}'
    local command=(env PYTHONPATH="$root/source/$source" "$py" -m vllm.entrypoints.openai.api_server
        --model "$target" --served-model-name qwen --max-model-len 32768
        --max-num-seqs 128 --max-num-batched-tokens 8192
        --no-enable-prefix-caching --disable-uvicorn-access-log
        --no-enable-flashinfer-autotune --gpu-memory-utilization 0.90
        --host 127.0.0.1 --port "$port" --speculative-config "$spec")
    printf '%q ' "${command[@]}" > "$dir/server-command.txt"
    date -u +%FT%TZ > "$dir/start-time.txt"
    setsid "${command[@]}" > "$dir/server.log" 2>&1 &
    server_pid=$!
    wait_ready
    date -u +%FT%TZ > "$dir/ready-time.txt"
    nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv -l 1 > "$dir/telemetry.csv" &
    monitor_pid=$!
    bash "$root/host_load.sh" "$dir/host-load.txt" &
    load_pid=$!
}

run_workload() {
    local dir=$1 workload=$2
    for c in "${concurrencies[@]}"; do run_cell "$dir" "$workload" "$c"; done
}

# Request budgets size each cell to roughly a minute of decoding.
requests_for() {
    case $1 in 1) echo 16 ;; 4) echo 64 ;; 8) echo 96 ;; 32) echo 256 ;; *) echo 64 ;; esac
}

run_cell() {
    local dir=$1 workload=$2 c=$3 cell=$1/$2/c$3 config=$1/$2/c$3/benchmark.yaml
    mkdir -p "$cell"
    sed "s|__PORT__|$port|g; s|__TARGET__|$target|g; s|__DATA__|$root/throughput_$2.jsonl|g; s|__FORMAT__|speed_bench_throughput_$2|g; s|__CONCURRENCY__|$3|g; s|__REQUESTS__|$(requests_for "$3")|g" "$root/sweep-fixed.yaml" > "$config"
    QUICK_ARTIFACTS=$cell/artifacts timeout 3600 "$client" profile --config "$config" > "$cell/client.log" 2>&1
}

wait_ready() {
    timeout 900 bash -c 'until curl -sf --max-time 1 "http://127.0.0.1:$1/health" > /dev/null; do kill -0 "$2" || exit 1; sleep 1; done' bash "$port" "$server_pid"
}

# Confirm the instrumented tree reaches the fused branch with this checkpoint.
probe_fusion() {
    local dir=$root/fusion-probe
    mkdir -p "$dir"
    start_server pr-instrumented union32 "$dir"
    grep -q FUSED_MARKOV_WALK_ACTIVE "$dir/server.log" && echo active > "$dir/fusion.txt" || echo inactive > "$dir/fusion.txt"
    stop_server
}

run_tests() {
    cp "$root/prsrc/tests/v1/spec_decode/test_dspark_topk.py" "$root/test_dspark_topk.py"
    ( cd "$root" && PYTHONPATH="$root/source/pr" "$py" -m pytest test_dspark_topk.py --confcutdir="$root" -q ) > tests.log 2>&1 && echo 0 > tests-exit.txt || echo $? > tests-exit.txt
}

stop_server() {
    if [[ -n $monitor_pid ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; monitor_pid=""; fi
    if [[ -n $load_pid ]]; then kill "$load_pid" 2>/dev/null || true; wait "$load_pid" 2>/dev/null || true; load_pid=""; fi
    [[ -n $server_pid ]] || return 0
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
}

finish() {
    stop_server
    echo "$1" > "$root/exit-code.txt"
}

main "$@"
