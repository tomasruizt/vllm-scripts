#!/usr/bin/env bash
# Queue this script inside canhazgpu, one persistent server for both policies.
set -euo pipefail
source "$(dirname "$0")/run-steady.sh"
mkdir -p "$root/onfullcg/repeat-1"
cd "$root"
export PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH"
export PYTHONPATH="$VLLM_CHECKOUT:$here"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
export MAX_NUM_SEQS=256 TENSOR_PARALLEL_SIZE=2 OMP_NUM_THREADS=1 FULL_CURVE_ROOT="$root"
printf '{"mode":"native","phase":"startup"}\n' > "$root/control.json"
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$root/gpus.txt"
trap 'code=$?; echo "$code" > "$root/exit-code.txt"; stop_server' EXIT
trap 'exit 143' TERM INT
start_server 8676 onfullcg "$root/onfullcg/repeat-1"
"$HOME/.venv/bin/python" "$here/run_av_full_sweep.py" "$root"
