#!/usr/bin/env bash
# Run each variant under `chg run` so its GPUs are released when it exits.
# PARALLEL=1 runs all variants at once (each on its own TP GPUs and port), otherwise one after another.
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/nfs-preprod-1/engine/hub_cache}
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/data/tomasruizt/cache
export TMPDIR=/data/tomasruizt/tmp
export VLLM_SERVER_DEV_MODE=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$HOME/code/vllm-dsv41/.venv/bin:$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
export TP=${TP:-8}
unset VLLM_USE_V2_MODEL_RUNNER
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
# chg only queues behind reservations, so also wait out unreserved processes holding GPU memory.
# WAIT_IDLE=0 skips this on a node shared only by chg reservations.
until [[ "${WAIT_IDLE:-1}" == 0 ]] || nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024 {busy=1} END {exit busy}'; do sleep 60; done
run_variant() {
    PORT=$((8810 + $2)) chg run --gpus "$TP" --wait 24h --note "DSV4.1-Flash spec-decode prefill cost: $1 TP$TP" -- \
        python3 run.py "$1" > "$1-job.log" 2>&1
}
RESULT=0
PIDS=()
i=0
VARIANTS=${VARIANTS:-nospec,k3,k5,k5av}
for variant in ${VARIANTS//,/ }; do
    if [[ "${PARALLEL:-0}" == 1 ]]; then run_variant "$variant" "$i" & PIDS+=("$!"); else run_variant "$variant" "$i" || RESULT=1; fi
    i=$((i + 1))
done
for pid in "${PIDS[@]}"; do wait "$pid" || RESULT=1; done
exit "$RESULT"
