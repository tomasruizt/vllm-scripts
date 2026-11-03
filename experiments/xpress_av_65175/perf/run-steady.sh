#!/usr/bin/env bash
# Steady-state AIPerf sweep of one XPress checkout: AV off vs AV on.
# Env: ROOT (run dir), REPO (checkout under test), PORTS, PROFILE_CONTEXT_LEN.
# AIPerf owns the concurrency grid and the trials (see steady-speedbench.yaml), so this
# script only starts a server per (GPU, mode), because the AV mode is a startup flag, and
# runs one AIPerf invocation against it. The two GPUs start with opposite modes, so mode
# and GPU effects cancel in the paired comparison.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${ROOT:?}
repo=${REPO:?}
py=${PY:-$HOME/code/vllm-dsv41/.venv/bin/python}
aiperf=${AIPERF:-$HOME/aiperf-venv/bin/aiperf}
ports=${PORTS:-"8640 8641"}
port_a=${ports%% *}
port_b=${ports##* }
modes=${MODES:-"off on"}

main() {
    if [[ ${1:-} == --worker ]]; then shift; worker "$@"; return; fi
    # launch-task.sh exits before the workers do (chg returns early), so signal
    # completion with a file written after both workers finish.
    trap 'printf "%s\n" "$?" > "$root/steady-exit-code.txt"' EXIT
    IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?one commit per reservation}"
    ((${#devices[@]} >= 1)) || { echo "need a GPU, got '$CUDA_VISIBLE_DEVICES'" >&2; exit 1; }
    mkdir -p "$root"
    [[ $here == "$root" ]] || cp "$here/run-steady.sh" "$here/steady-speedbench.yaml" "$root/"
    git -C "$repo" rev-parse HEAD > "$root/commit.txt"
    launch "${devices[0]}" "$port_a" "$modes" &
    if ((${#devices[@]} >= 2)); then launch "${devices[1]}" "$port_b" "$(reverse_modes "$modes")" & fi
    wait
}

reverse_modes() {  # other GPU starts with the other mode, so mode and GPU effects cancel
    local out="" mode
    for mode in $1; do out="$mode $out"; done
    echo -n "${out% }"
}

launch() {
    local gpu=$1 port=$2 plan=$3
    ROOT=$root REPO=$repo PORT=$port MODES="$modes" CUDA_VISIBLE_DEVICES=$gpu \
        bash "$root/run-steady.sh" --worker "$gpu" "$port" "$plan" \
        > "$root/worker-gpu$gpu.log" 2>&1
}

worker() {
    local gpu=$1 port=$2 plan=$3 mode server_pid=""
    export PATH="$HOME/aiperf-venv/bin:$HOME/.venv/bin:$PATH"
    export PYTHONPATH="$repo"
    export HF_HUB_CACHE=${HF_CACHE:-/mnt/nfs-preprod-1/engine/hub_cache}
    export XDG_CACHE_HOME=$HOME/cache TMPDIR=$HOME/tmp
    # The AV cost model profiles dummy steps at this context length (default 8192,
    # above our 4096 max model length). Match the measured workload.
    export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=${PROFILE_CONTEXT_LEN:-2048}
    cd "$root"  # never the checkout: `python -m` would shadow the patched copy
    trap "echo \$? > '$root/worker-gpu$gpu-exit-code.txt'; stop_server" EXIT
    trap 'exit 143' TERM INT
    for mode in $plan; do run_block "$port" "$mode"; done
}

run_block() {
    local port=$1 mode=$2 dir="$root/$2"
    mkdir -p "$dir"
    sed "s/__PORT__/$port/g" "$root/steady-speedbench.yaml" > "$dir/sweep.yaml"
    start_server "$port" "$mode" "$dir"
    export QUICK_ARTIFACTS="$dir/artifacts"
    timeout 5400 "$aiperf" profile --config "$dir/sweep.yaml" > "$dir/client.log" 2>&1
    stop_server
}

start_server() {
    local port=$1 mode=$2 dir=$3 av=false
    if curl --silent --max-time 2 "http://127.0.0.1:$port/health" > /dev/null; then
        echo "port $port already serves another server, aborting" >&2
        exit 1
    fi
    [[ "$mode" == on ]] && av=true
    spec='{"method":"dflash","model":"UIUC-SSAIL/Qwen3-8B-XPress-b16","num_speculative_tokens":15,"enable_adaptive_verification":'"$av"'}'
    local command=("$py" -m vllm.entrypoints.openai.api_server
        --model Qwen/Qwen3-8B --served-model-name qwen
        --max-model-len 16384 --max-num-seqs 256 --max-num-batched-tokens 8192
        --gpu-memory-utilization 0.90 --no-enable-prefix-caching --disable-uvicorn-access-log
        --host 127.0.0.1 --port "$port" --speculative-config "$spec")
    printf '%q ' "${command[@]}" > "$dir/server-command.txt"
    env | grep -E '^(VLLM|HF_|PYTHONPATH)=' | sort > "$dir/server-env.txt"
    echo "$CUDA_VISIBLE_DEVICES" > "$dir/gpu.txt"
    setsid "${command[@]}" > "$dir/server.log" 2>&1 &
    server_pid=$!
    for ((i = 0; i < 900; i++)); do
        kill -0 "$server_pid" 2>/dev/null || exit 1
        curl --silent --fail --max-time 1 "http://127.0.0.1:$port/health" > /dev/null && return
        sleep 1
    done
    exit 1
}

stop_server() {
    [[ -n "$server_pid" ]] || return 0
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
}

main "$@"
