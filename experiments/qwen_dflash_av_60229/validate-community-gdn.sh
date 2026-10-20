#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=1
root="$HOME/runs/qwen-gdn-community-20261019"
cd "$HOME/code/vllm-av-gdn-community"
export PYTHONPATH="$PWD"
"$HOME/.venv/bin/python" -m pytest tests/v1/worker/test_gpu_input_batch_v2.py tests/v1/spec_decode/test_adaptive_verification.py tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py tests/v1/cudagraph/test_cudagraph_dispatch.py tests/v1/cudagraph/test_cudagraph_manager.py tests/v1/attention/test_gdn_metadata_builder.py --override-ini addopts= -q > "$root/regression.log" 2>&1
"$HOME/.venv/bin/ruff" check vllm/v1/worker/gpu/input_batch.py vllm/v1/worker/gpu/model_runner.py tests/v1/worker/test_gpu_input_batch_v2.py > "$root/lint.log" 2>&1
"$HOME/.venv/bin/ruff" format --check vllm/v1/worker/gpu/input_batch.py vllm/v1/worker/gpu/model_runner.py tests/v1/worker/test_gpu_input_batch_v2.py >> "$root/lint.log" 2>&1
cd "$HOME/code/vllm-av-community-base"
export PYTHONPATH="$PWD"
test_file=tests/v1/worker/test_gpu_input_batch_v2.py
cp "$test_file" "$root/base-input-batch-test.py"
trap 'cp "$root/base-input-batch-test.py" "$test_file"' EXIT
cp "../vllm-av-gdn-community/$test_file" "$test_file"
if "$HOME/.venv/bin/python" -m pytest "$test_file" -k dummy_mamba --override-ini addopts= -q > "$root/negative-control.log" 2>&1; then
    echo "Negative control unexpectedly passed" >&2
    exit 1
fi
touch "$root/VALIDATED"
