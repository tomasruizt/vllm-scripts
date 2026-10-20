#!/usr/bin/env bash
# Torch-profile steps with a ~1K-token prefill, to split their ~85 ms into kernel time vs launch/CPU overhead.
# Scenarios: "solo" (one prompt, nothing else running) and "mixed" (one prompt joins 8 decoding requests).
# Usage (on the GPU host): ROOT=... chg run --gpus 1 -- bash profile-prefill-step.sh [torch|pyspy]
set -euo pipefail

root="${ROOT:-$HOME/runs/qwen-prefill-step-profile}"
port=8640
url="http://127.0.0.1:$port"
prompt_len=1080

main() {
    export PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH"
    export PYTHONPATH="$HOME/code/vllm-validate-60229"
    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
    mkdir -p "$root/traces"
    cd "$root"
    trap 'kill -TERM -- "-$server_pid" 2>/dev/null || true; echo $? > "$root/exit-code.txt"' EXIT
    start_server
    send 4 64  # warmup, outside the profile
    "run_${1:-torch}"
    echo DONE
}

run_torch() {
    profile solo profile_solo
    profile mixed profile_mixed
}

run_pyspy() {  # sample Python + native stacks of the engine core while it runs back-to-back solo prefills
    local pid spy
    pid=$(pgrep -s "$server_pid" -f EngineCore | head -1)
    py-spy record --pid "$pid" --threads --rate 200 --duration 30 --format raw -o pyspy-prefill.txt &
    spy=$!
    local end=$((SECONDS + 28))
    while ((SECONDS < end)); do send 1 1; done
    wait "$spy"
}

profile_solo() { send 1 16; }

profile_mixed() {
    send 8 400 &
    local bg=$!
    sleep 3  # background requests are past their prefill and decoding
    send 1 16
    wait "$bg"  # a bare `wait` would also wait on the server
}

profile() {  # profile <name> <fn>: traces of one start/stop window go to traces/<name>
    local name=$1 fn=$2
    curl -sf -X POST "$url/start_profile"
    "$fn"
    curl -sf -X POST "$url/stop_profile"
    mkdir -p "traces/$name"
    find traces -maxdepth 1 -type f -exec mv {} "traces/$name/" \;
    echo "PROFILED $name"
}

send() {  # send <n> <max_tokens>: n concurrent requests with a fixed-length random-token prompt
    local n=$1 max_tokens=$2 i pids=()
    for ((i=0; i<n; i++)); do
        python -c "import json,random; print(json.dumps({'model':'qwen','prompt':[random.randint(1000,100000) for _ in range($prompt_len)],'max_tokens':$max_tokens,'min_tokens':$max_tokens,'temperature':0}))" \
            | curl -sf "$url/v1/completions" -H 'Content-Type: application/json' -d @- > /dev/null &
        pids+=($!)
    done
    wait "${pids[@]}"
}

start_server() {
    local spec='{"method":"dflash","model":"z-lab/Qwen3.5-35B-A3B-DFlash","revision":"52cb554b4995dede3e2e1bdb129cdb1f3529332b","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}'
    local prof='{"profiler":"torch","torch_profiler_dir":"'"$root/traces"'","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true}'
    local command=(python -m vllm.entrypoints.openai.api_server
        --model /data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
        --served-model-name qwen --language-model-only --max-model-len 16384
        --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92
        --no-enable-prefix-caching --disable-uvicorn-access-log
        --host 127.0.0.1 --port "$port" --speculative-config "$spec" --profiler-config "$prof")
    printf '%q ' "${command[@]}" > server-command.txt
    setsid "${command[@]}" > server.log 2>&1 &
    server_pid=$!
    until curl -sf --max-time 1 "$url/health" > /dev/null; do kill -0 "$server_pid"; sleep 2; done
}

main "$@"
