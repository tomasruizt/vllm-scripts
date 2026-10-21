#!/usr/bin/env bash
# Only server orchestration lives here. AIPerf owns the measured sweep.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/run-steady.sh"

main_sweep() {
    local output=${1:?output directory}
    local worker=${2:?worker a or b}
    local port=${3:?port}
    root="$output/worker-$worker"
    mkdir -p "$root"
    export PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH"
    export OMP_NUM_THREADS=1 TENSOR_PARALLEL_SIZE=2 MAX_NUM_SEQS=256
    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
    cd "$root"
    trap 'echo $? > "$root/exit-code.txt"; stop_server' EXIT
    trap 'exit 143' TERM INT
    printf '%s\n' "${CUDA_VISIBLE_DEVICES:?GPU reservation required}" > "$root/allocated-gpus.txt"
    cp "$here/run-profiling-sweep.sh" "$here/profiling-sweep.yaml" "$root/"
    if [[ "$worker" == a ]]; then
        run_arm main "$HOME/code/vllm-av-main-gdn-support" "$port"
        run_arm community "$HOME/code/vllm-av-community-base" "$port"
        run_arm gdn "$HOME/code/vllm-av-gdn-community" "$port"
    else
        run_arm gdn "$HOME/code/vllm-av-gdn-community" "$port"
        run_arm community "$HOME/code/vllm-av-community-base" "$port"
        run_arm main "$HOME/code/vllm-av-main-gdn-support" "$port"
    fi
    echo DONE
}

run_arm() {
    local arm=$1 checkout=$2 port=$3
    local dir="$root/on-$arm"
    mkdir -p "$dir"
    export PYTHONPATH="$checkout"
    export QUICK_ARTIFACTS="$dir/artifacts"
    sed "s/__PORT__/$port/g" "$here/profiling-sweep.yaml" > "$dir/sweep.yaml"
    "$aiperf" config validate "$dir/sweep.yaml" > "$dir/validation.log" 2>&1
    start_server "$port" "on-$arm" "$dir"
    echo "START $arm"
    timeout 2400 "$aiperf" profile --config "$dir/sweep.yaml" > "$dir/client.log" 2>&1
    echo "DONE $arm"
    stop_server
}

main_sweep "$@"
