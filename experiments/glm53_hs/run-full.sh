#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
export VLLM_USE_V2_MODEL_RUNNER=${1:?Specify runner 0 or 1}
export HS_OUTPUT_DIR=$HS_RUNS_DIR/runner${VLLM_USE_V2_MODEL_RUNNER}
mkdir -p "$HS_OUTPUT_DIR"
cd "$VLLM_DIR"
exec canhazgpu run --gpus "$TP_SIZE" --timeout 2h --note "glm-hs-pr59037-full-runner$VLLM_USE_V2_MODEL_RUNNER" -- /home/tomasruizt/.venv/bin/python "$SCRIPT_DIR/full_glm_hs.py"
