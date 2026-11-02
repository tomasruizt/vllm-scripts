#!/usr/bin/env bash
# Steady-state AIPerf sweep of one XPress checkout: AV off vs AV on, three repeats.
# Two GPUs alternate the mode per repeat, so each GPU measures both modes and GPU
# differences cancel in the paired comparison. A fresh server per repeat.
# Env: ROOT (run dir), REPO (checkout under test), CELLS, PORTS, PROFILE_CONTEXT_LEN.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${ROOT:?}
repo=${REPO:?}
py=${PY:-$HOME/code/vllm-dsv41/.venv/bin/python}
aiperf=${AIPERF:-$HOME/aiperf-venv/bin/aiperf}
ports=${PORTS:-"8640 8641"}
port_a=${ports%% *}
port_b=${ports##* }
# c:requests. Sized for ~30 s per cell and >= 3 cycles of the concurrency. The
# trailing c1 is a drift sentinel. c128 is the top valid cell: 294,912 KV tokens
# fit ~170 of these ~1.7k-token requests, so c256 would queue and fail the gates.
cells=${CELLS:-1:24 8:48 32:96 64:192 128:384 1:24}

main() {
    if [[ ${1:-} == --worker ]]; then shift; worker "$@"; return; fi
    # launch-task.sh exits before the workers do (chg returns early), so signal
    # completion with a file written after both workers finish.
    trap 'printf "%s\n" "$?" > "$root/steady-exit-code.txt"' EXIT
    local repeats=${1:-3}
    IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?one commit per reservation}"
    ((${#devices[@]} >= 2)) || { echo "need 2 GPUs, got '$CUDA_VISIBLE_DEVICES'" >&2; exit 1; }
    mkdir -p "$root"
    [[ $here == "$root" ]] || cp "$here/run-steady.sh" "$here/steady-speedbench.yaml" "$root/"
    echo "$cells" > "$root/cells.txt"
    git -C "$repo" rev-parse HEAD > "$root/commit.txt"
    launch "${devices[0]}" "$port_a" "$(plan off on "$repeats")" &
    launch "${devices[1]}" "$port_b" "$(plan on off "$repeats")" &
    wait
}

launch() {
    local gpu=$1 port=$2 plan=$3
    ROOT=$root REPO=$repo CELLS=$cells PORT=$port CUDA_VISIBLE_DEVICES=$gpu \
        bash "$root/run-steady.sh" --worker "$gpu" "$port" "$plan" \
        > "$root/worker-gpu$gpu.log" 2>&1
}

plan() {  # "1:off 2:on 3:off ..."
    local first=$1 second=$2 repeats=$3 r
    for ((r = 1; r <= repeats; r++)); do
        ((r % 2)) && echo -n "$r:$first " || echo -n "$r:$second "
    done
}

worker() {
    local gpu=$1 port=$2 plan=$3 item server_pid=""
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
    for item in $plan; do run_repeat "$gpu" "$port" "${item%%:*}" "${item#*:}"; done
}

run_repeat() {
    local gpu=$1 port=$2 repeat=$3 mode=$4 cell
    local dir="$root/$mode/repeat-$repeat"
    mkdir -p "$dir"
    echo "$gpu" > "$dir/gpu.txt"
    start_server "$port" "$mode" "$dir"
    for cell in $cells; do run_cell "$port" "$dir" "${cell%%:*}" "${cell#*:}"; done
    stop_server
}

run_cell() {
    local port=$1 dir=$2 c=$3 n=$4 name="c$3"
    [[ -d "$dir/artifacts-$name" ]] && name="$name-end"
    local config="$dir/$name.yaml"
    sed "s/__PORT__/$port/g; s/__C__/$c/; s/__N__/$n/" "$root/steady-speedbench.yaml" > "$config"
    export QUICK_ARTIFACTS="$dir/artifacts-$name"
    echo "START $dir $name n=$n"
    timeout 1800 "$aiperf" profile --config "$config" > "$dir/client-$name.log" 2>&1
    echo "DONE $dir $name"
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
        --max-model-len 4096 --max-num-seqs 256 --max-num-batched-tokens 8192
        --gpu-memory-utilization 0.90 --no-enable-prefix-caching --disable-uvicorn-access-log
        --host 127.0.0.1 --port "$port" --speculative-config "$spec")
    printf '%q ' "${command[@]}" | tee "$dir/server-command.txt" > "$root/$mode/server-command.txt"
    env | grep -E '^(VLLM|HF_|PYTHONPATH)=' | sort > "$dir/server-env.txt"
    "$py" -c "import vllm; print(vllm.__file__)" > "$dir/vllm-path.txt"
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
