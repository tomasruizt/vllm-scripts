#!/usr/bin/env bash
# Steady-state AV off/on sweep. Two workers on two GPUs alternate the AV mode per repeat,
# so each GPU measures both modes and GPU differences cancel in the paired comparison.
# Usage: run-steady.sh <gpu_a> <gpu_b> [repeats]. Env: ROOT, CELLS.
# Optional checkouts: VLLM_CHECKOUT, VLLM_DIAG_CHECKOUT. Inject curves with VLLM_AV_COST_CURVES.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root="${ROOT:-$HOME/runs/qwen-speedbench-steady-20261016}"
# c:requests. Sized for ~30s per cell and >= 3 cycles of the concurrency. Trailing c1 is the drift sentinel.
# Top level is c128: with 0.92 GPU memory the hybrid KV cache holds ~150 seqs, so c256 queues and preempts.
CELLS="${CELLS:-1:24 8:96 32:192 64:320 128:512 1:24}"
aiperf=$HOME/aiperf-venv/bin/aiperf

main() {
    if [[ "${1:-}" == --worker ]]; then shift; worker "$@"; exit; fi
    local gpu_a=$1 gpu_b=$2 repeats=${3:-3}
    mkdir -p "$root"
    cp "$here/run-steady.sh" "$here/steady-speedbench.yaml" "$root/"
    echo "$CELLS" > "$root/cells.txt"
    launch "$gpu_a" "${PORT_A:-8630}" "${PLAN_A:-$(plan off on "$repeats")}" &
    launch "$gpu_b" "${PORT_B:-8631}" "${PLAN_B:-$(plan on off "$repeats")}" &
    wait
}

plan() {  # "1:off 2:on 3:off ..."
    local first=$1 second=$2 repeats=$3 r
    for ((r=1; r<=repeats; r++)); do (( r % 2 )) && echo -n "$r:$first " || echo -n "$r:$second "; done
}

launch() {
    canhazgpu run --gpu-ids "$1" --timeout 3h --note "Qwen DFlash AV steady sweep" -- \
        env ROOT="$root" CELLS="$CELLS" bash "$root/run-steady.sh" --worker "$@" > "$root/worker-gpu$1.log" 2>&1
}

worker() {
    local gpu=$1 port=$2 plan=$3 item
    rm -f "$root/worker-gpu$gpu-exit-code.txt"
    export PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH"
    export PYTHONPATH="${VLLM_CHECKOUT:-$HOME/code/vllm-validate-60229}"
    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
    cd "$HOME/code/vllm-validate-60229"
    git diff --exit-code HEAD
    git rev-parse HEAD > "$root/commit.txt"
    # Leave the checkout: `python -m` puts the cwd ahead of PYTHONPATH, which would shadow patched copies.
    cd "$root"
    trap "echo \$? > '$root/worker-gpu$gpu-exit-code.txt'; stop_server" EXIT
    trap 'exit 143' TERM INT
    for item in $plan; do run_repeat "$gpu" "$port" "${item%%:*}" "${item#*:}"; done
}

server_pid=""

run_repeat() {
    local gpu=$1 port=$2 repeat=$3 mode=$4 cell
    local dir="$root/$mode/repeat-$repeat"
    mkdir -p "$dir"
    echo "$gpu" > "$dir/gpu.txt"
    start_server "$port" "$mode" "$dir"
    for cell in $CELLS; do run_cell "$port" "$dir" "${cell%%:*}" "${cell#*:}"; done
    stop_server
}

run_cell() {
    local port=$1 dir=$2 c=$3 n=$4 name="c$3"
    [[ -d "$dir/artifacts-$name" ]] && name="$name-end"
    local config="$dir/$name.yaml"
    sed "s/__PORT__/$port/g; s/__C__/$c/; s/__N__/$n/" "$root/steady-speedbench.yaml" > "$config"
    export QUICK_ARTIFACTS="$dir/artifacts-$name"
    echo "START $dir $name n=$n"
    timeout 900 "$aiperf" profile --config "$config" > "$dir/client-$name.log" 2>&1
    echo "DONE $dir $name"
}

start_server() {
    local port=$1 mode=$2 dir=$3 av=false k=7 spec i env=(PYTHONPATH="$PYTHONPATH")
    # Refuse to share a port: the health check would pass against another run's server.
    if curl --silent --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
        echo "port $port already serves another server, aborting" >&2
        exit 1
    fi
    [[ "$mode" == on* ]] && av=true
    # offk2*: AV off with only 2 drafts, same step shape as AV on at c128 (control for AV overhead).
    [[ "$mode" =~ k([0-9]+) ]] && k="${BASH_REMATCH[1]}"
    # onpf: AV on, but verify all drafts in steps with prefill (av-full-verify-on-prefill.patch).
    [[ "$mode" == onpf ]] && env=(PYTHONPATH="$HOME/code/vllm-av-pf" VLLM_AV_FULL_VERIFY_ON_PREFILL=1)
    # *diag*: per-step AV trace (av-diagnostics.patch + av_trace.py), combinable with kN/cg.
    [[ "$mode" == *diag* ]] && env=(PYTHONPATH="${VLLM_DIAG_CHECKOUT:-$HOME/code/vllm-av-diag}" AV_TRACE_PATH="$dir/av-trace.jsonl")
    # Hold the drafter at k=7 while controlling AV's total decode verification budget.
    [[ "$mode" =~ cap([0-9]+) ]] && env+=(VLLM_AV_VERIFY_BUDGET_PER_REQ="${BASH_REMATCH[1]}")
    [[ "$mode" != *cost* ]] && env+=(VLLM_AV_COST_CURVES=)
    [[ "$mode" == *cost* && -n "${VLLM_AV_COST_CURVES:-}" ]] && env+=(VLLM_AV_COST_CURVES="$VLLM_AV_COST_CURVES")
    spec='{"method":"dflash","model":"z-lab/Qwen3.5-35B-A3B-DFlash","revision":"52cb554b4995dede3e2e1bdb129cdb1f3529332b","num_speculative_tokens":'"$k"',"draft_sample_method":"probabilistic","enable_adaptive_verification":'"$av"'}'
    # max-num-batched-tokens >> max-num-seqs * (k+1), so verification never starves prefill.
    local command=("$HOME/.venv/bin/python" -m vllm.entrypoints.openai.api_server
        --model /data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
        --served-model-name qwen --language-model-only --max-model-len 16384
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
        --max-num-seqs "${MAX_NUM_SEQS:-512}" --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92
        --no-enable-prefix-caching --disable-uvicorn-access-log
        --host 127.0.0.1 --port "$port" --speculative-config "$spec")
    # *cg: the default capture list (stops at 1,024 tokens) plus prefill-sized batches, so prefill steps get piecewise graphs.
    [[ "$mode" == *cg* ]] && command+=(--compilation-config '{"cudagraph_capture_sizes":['"$(cg_sizes)"']}')
    printf '%q ' "${command[@]}" | tee "$dir/server-command.txt" > "$root/$mode/server-command.txt"
    echo "${env[*]}" > "$dir/server-env.txt"
    env "${env[@]}" "$HOME/.venv/bin/python" -c "import vllm; print(vllm.__file__)" > "$dir/vllm-path.txt"
    setsid env "${env[@]}" "${command[@]}" > "$dir/server.log" 2>&1 &
    server_pid=$!
    export -f watch_server
    setsid bash -c 'watch_server "$@"' _ "$$" "$server_pid" > "$dir/watchdog.log" 2>&1 &
    for ((i=0; i<600; i++)); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "FAILED server startup: $dir" >&2
            tail -n 20 "$dir/server.log" >&2
            exit 1
        fi
        curl --silent --fail --max-time 1 "http://127.0.0.1:$port/health" >/dev/null && return
        sleep 1
    done
    exit 1
}

cg_sizes() {
    echo 1 2 4 $(seq 8 8 248) $(seq 256 16 1024) $(seq 1280 256 4096) 5120 6144 | tr ' ' ','
}

stop_server() {
    [[ -n "$server_pid" ]] || return 0
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    timeout 30 tail --pid="$server_pid" -f /dev/null || kill -KILL -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
}

watch_server() {
    local owner=$1 server=$2
    while kill -0 "$owner" 2>/dev/null && kill -0 "$server" 2>/dev/null; do sleep 1; done
    kill -0 "$server" 2>/dev/null || return 0
    kill -TERM -- "-$server" 2>/dev/null || true
    timeout 30 tail --pid="$server" -f /dev/null || kill -KILL -- "-$server" 2>/dev/null || true
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
