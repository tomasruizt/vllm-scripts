#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data/tomasruizt/runs/capture-worst-case-20261020}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export ROOT
export PYTHONPATH=/data/tomasruizt/runtime/vllm-capture-grid-58b3298457
export HF_HUB_CACHE=/data/tomasruizt/hub_cache
export XDG_CACHE_HOME=/data/tomasruizt/cache
export TMPDIR=/data/tomasruizt/tmp
export VLLM_SERVER_DEV_MODE=1
export EXPERIMENT_PYTHON=/data/tomasruizt/capture-env/bin/python
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="/data/tomasruizt/capture-env/bin:$HOME/.venv/bin:$CUDA_HOME/bin:$PATH"
unset VLLM_USE_V2_MODEL_RUNNER
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
timeout 7200 bash -c 'until test -f prepare-exit-code.txt; do sleep 5; done'
[[ $(<prepare-exit-code.txt) == 0 ]]
chg run --gpus 1 --wait 30m --timeout 75m --note 'H100 capture-grid dense32B worst-case pilot' -- "$EXPERIMENT_PYTHON" run.py dense > dense-job.log 2>&1 &
FIRST_PID=$!
chg run --gpus 1 --wait 30m --timeout 75m --note 'H100 capture-grid GDN27B worst-case pilot' -- "$EXPERIMENT_PYTHON" run.py gdn > gdn-job.log 2>&1 &
SECOND_PID=$!
trap 'kill -TERM "$FIRST_PID" "$SECOND_PID" 2>/dev/null || true; exit 143' TERM INT
RESULT=0
wait "$FIRST_PID" || RESULT=1
wait "$SECOND_PID" || RESULT=1
"$EXPERIMENT_PYTHON" collect.py "$ROOT"
exit "$RESULT"
