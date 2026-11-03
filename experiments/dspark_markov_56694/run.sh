#!/usr/bin/env bash
set -euo pipefail
root=${ROOT:?}
runtime=${RUNTIME:?set RUNTIME to the precompiled vLLM checkout}
py=${PY:-$runtime/.venv/bin/python}
client=$root/client-venv/bin/aiperf
port=${PORT:-8664}
server_pid=""
client_pid=""
monitor_pid=""
export HF_HUB_CACHE=${HF_HUB_CACHE:?set HF_HUB_CACHE to the shared cache}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$runtime/.venv/bin:$CUDA_HOME/bin:$PATH
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
    export PYTHONPATH=$root/source/pr
    cp "$root/source/pr/tests/v1/spec_decode/test_dspark_topk.py" "$root/test_dspark_topk.py"
    "$py" -m pytest "$root/test_dspark_topk.py" --confcutdir="$root" -q > tests.log 2>&1 && echo 0 > tests-exit.txt || echo $? > tests-exit.txt
    for mode in ${MODES:-dense base32 union32}; do run_mode "$mode"; done
    touch DONE
}

run_mode() {
    local mode=$1
    local source=pr
    case $mode in dense|base32|union32) ;; *) echo "Unknown mode: $mode" >&2; return 1 ;; esac
    [[ $mode == base32 ]] && source=base
    export PYTHONPATH=$root/source/$source
    local dir=$root/$mode
    mkdir "$dir"
    if curl -sf --max-time 2 "http://127.0.0.1:$port/health" > /dev/null; then
        echo "Port $port already serves another server" >&2
        return 1
    fi
    local spec='{"method":"dspark","model":"'"$draft"'","num_speculative_tokens":8}'
    [[ $mode != dense ]] && spec=${spec%\}}',"dspark_draft_topk":32}'
    local command=("$py" -m vllm.entrypoints.openai.api_server
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
    "$py" "$root/check_outputs.py" "$port" "$root/throughput_8k.jsonl" "$dir/outputs.json"
    nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv -l 1 > "$dir/telemetry.csv" &
    monitor_pid=$!
    for workload in ${WORKLOADS:-8k 16k}; do
        local config=$dir/$workload.yaml
        sed "s|__PORT__|$port|g; s|__TARGET__|$target|g; s|__DATA__|$root/throughput_$workload.jsonl|g; s|__FORMAT__|speed_bench_throughput_$workload|g" "$root/sweep.yaml" > "$config"
        export QUICK_ARTIFACTS=$dir/$workload/artifacts
        mkdir -p "$dir/$workload"
        setsid timeout 3600 "$client" profile --config "$config" > "$dir/$workload/client.log" 2>&1 &
        client_pid=$!
        wait "$client_pid"
        client_pid=""
    done
    kill "$monitor_pid" || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
    stop_server
}

finish() {
    local code=$1
    stop_group "$client_pid"
    if [[ -n $monitor_pid ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    stop_server
    echo "$code" > "$root/exit-code.txt"
}

wait_ready() {
    timeout 900 bash -c 'until curl -sf --max-time 1 "http://127.0.0.1:$1/health" > /dev/null; do kill -0 "$2" || exit 1; sleep 1; done' bash "$port" "$server_pid"
}

stop_server() {
    stop_group "$server_pid"
    server_pid=""
}

stop_group() {
    [[ -n $1 ]] || return 0
    kill -TERM -- "-$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
}

main "$@"
