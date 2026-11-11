#!/usr/bin/env bash
# Torch-profile 3 s of the steady state at concurrency C of main and the PR on the same GPUs,
# to compare GPU time and host gaps per engine step. Traces land in <variant>/trace.
# Usage, on the AMD host from the run dir: chg run --gpu-ids A,B,C,D -- bash amd_prof.sh [C]
set -uo pipefail
concurrency=${1:-1}
export REPEATS=1 CONCURRENCIES=$concurrency DURATION=60 GSM8K_QUESTIONS=0 AIME_N=0

main() {
    local build
    for build in main hier; do profile "$build-mtp5-prof-c$concurrency"; done
    echo PROF_DONE
}

profile() {
    local variant=$1 name=tr-prof-$1
    local config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/work/$variant/trace\",\"torch_profiler_with_stack\":false}"
    PODMAN_NAME=$name EXTRA_SERVER_ARGS="--profiler-config $config" bash amd_e2e.sh "$variant" > "$variant.out" 2>&1 &
    local job=$!
    until grep -q "CELL $variant" "$variant.out"; do kill -0 $job 2>/dev/null || { echo "FAILED $variant"; return 1; }; sleep 5; done
    sleep 25  # steady state
    local port
    port=$(grep -o -- '--port [0-9]*' "$variant/server-command.txt" | awk '{print $2}')
    podman exec "$name" curl -sf -X POST "http://127.0.0.1:$port/start_profile"
    sleep 3
    podman exec "$name" curl -sf -X POST "http://127.0.0.1:$port/stop_profile"
    echo "PROFILED $variant $(date -u +%FT%TZ)"
    wait $job
}

main
