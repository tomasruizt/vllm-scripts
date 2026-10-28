#!/usr/bin/env bash
# Run all variants at once, each on one GPU under `chg run` (released when it exits).
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export HF_HUB_CACHE=/mnt/nfs-preprod-1/engine/hub_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$HOME/cache
export TMPDIR=$HOME/tmp
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$HOME/code/vllm-dsv41/.venv/bin:$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
unset VLLM_USE_V2_MODEL_RUNNER
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
PIDS=()
i=0
VARIANTS=${VARIANTS:-nospec,k5,k5av}
for variant in ${VARIANTS//,/ }; do
    PORT=$((${PORT_BASE:-8850} + i)) chg run --gpus 1 --wait 2h --note "DSV4 small AV prefill profile: $variant" -- \
        python3 run.py "$variant" > "$variant-job.log" 2>&1 &
    PIDS+=("$!")
    i=$((i + 1))
done
RESULT=0
for pid in "${PIDS[@]}"; do wait "$pid" || RESULT=1; done
exit "$RESULT"
