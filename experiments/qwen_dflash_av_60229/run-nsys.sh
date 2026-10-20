#!/usr/bin/env bash
# Profile 3 s of steady c128 decode with nsys, AV on or off, to measure AV's per-step overhead.
# Usage: run-nsys.sh <gpu> <port> <on|off> <out_dir>
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
aiperf=$HOME/aiperf-venv/bin/aiperf
model=/data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754

main() {
    if [[ "${1:-}" == --inner ]]; then shift; inner "$@"; exit; fi
    mkdir -p "$4"
    canhazgpu run --gpu-ids "$1" --timeout 1h --note "AV nsys profile" -- bash "$here/run-nsys.sh" --inner "$@" > "$4/run.log" 2>&1
}

inner() {
    local port=$2 mode=$3 out=$4
    export PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH"
    export PYTHONPATH="$HOME/code/vllm-validate-60229"
    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
    cd "$out"  # never run from the checkout: python -m would shadow PYTHONPATH
    export TMPDIR="$out/tmp"  # /tmp/nvidia belongs to another user, nsys cannot write there
    mkdir -p "$TMPDIR"
    start_server "$port" "$mode" "$out"
    trap 'kill -TERM -- "-$server_pid" 2>/dev/null || true' EXIT
    sed "s/__PORT__/$port/g; s/__C__/128/; s/__N__/768/" "$here/steady-speedbench.yaml" > c128.yaml
    QUICK_ARTIFACTS="$out/artifacts-c128" "$aiperf" profile --config c128.yaml > client.log 2>&1 &
    local client=$!
    wait_running "$port" 120
    curl -s -X POST "http://127.0.0.1:$port/start_profile" > /dev/null
    sleep 3  # profiling window
    curl -s -X POST "http://127.0.0.1:$port/stop_profile" > /dev/null
    wait "$client"
    echo DONE
}

start_server() {
    local port=$1 mode=$2 out=$3 av=false
    [[ "$mode" == on ]] && av=true
    local spec='{"method":"dflash","model":"z-lab/Qwen3.5-35B-A3B-DFlash","revision":"52cb554b4995dede3e2e1bdb129cdb1f3529332b","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":'"$av"'}'
    local nsys=(nsys profile -o "$out/av-$mode-c128" --force-overwrite=true --trace=cuda,nvtx,osrt
        --cuda-graph-trace=node --python-sampling=true --trace-fork-before-exec=true
        --capture-range=cudaProfilerApi --capture-range-end=stop)
    setsid "${nsys[@]}" "$HOME/.venv/bin/python" -m vllm.entrypoints.openai.api_server --model "$model" \
        --served-model-name qwen --language-model-only --max-model-len 16384 \
        --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 \
        --no-enable-prefix-caching --disable-uvicorn-access-log --host 127.0.0.1 --port "$port" \
        --speculative-config "$spec" --profiler-config '{"profiler":"cuda"}' > server.log 2>&1 &
    server_pid=$!
    timeout 900 bash -c "until curl -sf --max-time 1 http://127.0.0.1:$port/health >/dev/null; do sleep 2; done"
}

wait_running() {  # block until the server runs at least $2 requests
    timeout 300 bash -c "until [ \"\$(curl -s http://127.0.0.1:$1/metrics | awk '/^vllm:num_requests_running/ {print int(\$2)}')\" -ge $2 ] 2>/dev/null; do sleep 1; done"
}

main "$@"
