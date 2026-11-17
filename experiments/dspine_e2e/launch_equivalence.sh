#!/usr/bin/env bash
# PR head vs the dspine-refactor loading, one after another on the same GPU.
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
TREE=${TREE:-$HOME/code/vllm-dspine-new}
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$TREE/.venv/bin:$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1
unset VLLM_USE_V2_MODEL_RUNNER
REV=260a6c23070ca873ff1a5f33b7119ac3f6a245ee
run_all() {
    set -e
    git -C "$TREE" checkout -q pr-67094-new
    python equivalence.py --draft-revision $REV --out pr-head.json > pr-head.log 2>&1
    git -C "$TREE" checkout -q dspine-refactor
    python equivalence.py --draft-revision $REV --out refactor.json > refactor.log 2>&1
}
export -f run_all
export TREE REV
chg run --gpus 1 --wait 1h --timeout 2h --note "DSpine loading refactor equivalence" -- bash -c run_all
