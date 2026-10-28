#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
trap 'printf "%s\n" "$?" > job-exit-code.txt' EXIT
export PYTHONPATH="$ROOT:$HOME/code/vllm-qwen38-capture-grid"
export VLLM_SERVER_DEV_MODE=1
export HF_HUB_CACHE=/data/engine/hub_cache
export PATH="$HOME/.venv/bin:$HOME/.local/bin:$PATH"
unset VLLM_USE_V2_MODEL_RUNNER
if [[ ${SMOKE:-0} == 1 ]]; then
    export HF_HUB_CACHE=/data/tomasruizt/hub_cache
    export MODEL
    MODEL=$("$HOME/.venv/bin/hf" download inference-optimization/GLM-5.3-Flash-2.6B-A2.3B-small --revision a2d7ca22e78d7961d47ce820af0c37e67a62a657 --cache-dir "$HF_HUB_CACHE" --quiet)
fi
"$HOME/.venv/bin/python" -c 'from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager; from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture; assert callable(ModelCudaGraphManager.run_pw_graph); assert callable(BreakableCUDAGraphCapture.replay)'
chg run --gpus "${TP:-4}" --wait 3h --timeout "${RUN_TIMEOUT:-30m}" --note 'GLM5.3 Flash graph-cost probe' -- "$HOME/.venv/bin/python" run_cost_probe.py
