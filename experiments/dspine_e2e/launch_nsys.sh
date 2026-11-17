#!/usr/bin/env bash
# nsys kernel breakdown (per NVTX range: execute/sample/draft) of one variant.
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export CUDA_HOME=/usr/local/cuda-13.0
TREE=${TREE:-$HOME/code/vllm-dspine}
export PATH="$TREE/.venv/bin:$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1
unset VLLM_USE_V2_MODEL_RUNNER
variant=${VARIANT:-ref}
export VLLM_DSPINE_SCORING=${variant%_tf32}
chg run --gpus 1 --wait 1h --timeout 1h --note "DSpine nsys: $variant" -- \
    nsys profile -o "nsys-$variant" -f true --capture-range=cudaProfilerApi \
    --capture-range-end=stop --cuda-graph-trace=node -t cuda,nvtx \
    python profile_steps.py --nsys --out "steps-$variant.jsonl" $PROFILE_ARGS \
    > "nsys-$variant.log" 2>&1
nsys stats -q --report nvtx_kern_sum --format csv -o "nsys-$variant" "nsys-$variant.nsys-rep" \
    > /dev/null 2>&1 || true
nsys stats -q --report cuda_gpu_kern_sum --format csv -o "nsys-$variant" "nsys-$variant.nsys-rep" \
    > /dev/null 2>&1 || true
