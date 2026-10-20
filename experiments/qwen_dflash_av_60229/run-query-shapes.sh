#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.venv/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/runs/qwen-dflash-av-60229:$HOME/code/vllm-validate-60229"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
export MODEL=/data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
export PROBE_OUTPUT="${PROBE_OUTPUT:-$HOME/runs/qwen-q1q2-20261019}"
cd "$HOME/runs/qwen-dflash-av-60229"
"$HOME/.venv/bin/python" -c 'import vllm; print(vllm.__file__, flush=True)'
"$HOME/.venv/bin/python" probe_query_shapes.py
