#!/usr/bin/env bash
# Run ROUNDS one after another, the variants of a round in parallel, each under `chg run` on its own TP GPUs and port.
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
# The login environment sets its own HF_HUB_CACHE, so override it rather than defaulting it.
export HF_HUB_CACHE=${MODEL_CACHE:-/data/engine/hub_cache}
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/data/tomasruizt/cache
export TMPDIR=/data/tomasruizt/tmp
export VLLM_SERVER_DEV_MODE=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$HOME/code/vllm-dsv41/.venv/bin:$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
export TP=${TP:-2}
unset VLLM_USE_V2_MODEL_RUNNER
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
ROUNDS=${ROUNDS:-base-full,pr-full,eager/base-av,pr-av}

run_variant() {
    PORT=$((8820 + $2)) chg run --gpus "$TP" --wait 24h --note "PR #59986 DSV4.1 FULL cudagraph validation: $1 TP$TP" -- \
        python3 run.py "$1" > "$1-job.log" 2>&1
}

run_round() {
    local pids=() i=0 result=0
    for variant in ${1//,/ }; do
        run_variant "$variant" "$i" & pids+=("$!")
        i=$((i + 1))
    done
    for pid in "${pids[@]}"; do wait "$pid" || result=1; done
    return "$result"
}

RESULT=0
for round in ${ROUNDS//\// }; do run_round "$round" || RESULT=1; done
exit "$RESULT"
