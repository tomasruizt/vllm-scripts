#!/usr/bin/env bash
# Run the scoring microbenchmark on one GPU under `chg run` (released when it exits).
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export PATH="${VENV:-$HOME/code/vllm-main-191e5d5/.venv}/bin:$PATH"
export TORCHINDUCTOR_CACHE_DIR=$HOME/cache/inductor
export BENCH_ARGS=${BENCH_ARGS:-}
chg run --gpus 1 --wait 1h --timeout 30m --note "DSpine scoring microbench" -- \
    bash -c 'nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader; python bench.py --out results.csv $BENCH_ARGS' > bench.log 2>&1
