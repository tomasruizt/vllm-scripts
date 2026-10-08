#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
export VLLM_USE_V2_MODEL_RUNNER=${1:?Specify runner 0 or 1}
export ORACLE_SMALL=${ORACLE_SMALL:-0}
export HS_OUTPUT_DIR=${HS_OUTPUT_DIR:-$HS_RUNS_DIR/same-pass-runner${VLLM_USE_V2_MODEL_RUNNER}}
cd "$VLLM_DIR"
count=$TP_SIZE
if [[ $ORACLE_SMALL == 1 ]]; then count=1; fi
exec canhazgpu run --gpus "$count" --wait 30m --timeout 1h --note "glm-hs-same-pass-runner$VLLM_USE_V2_MODEL_RUNNER" -- /home/tomasruizt/.venv/bin/python "$SCRIPT_DIR/validate_same_pass.py"
