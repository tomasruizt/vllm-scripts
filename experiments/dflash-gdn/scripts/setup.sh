#!/usr/bin/env bash
set -euo pipefail
engine=${1:?Usage: setup.sh ENGINE ENV_DIR [VLLM_CHECKOUT]}
env_dir=${2:?Specify a new environment directory}
uv venv --python 3.12 "$env_dir"
case "$engine" in
  vllm)
    checkout=${3:?Specify the checked-out vLLM source directory}
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=db7f1f671419535b85fa73e29ff061e287e39459 \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
      uv pip install --python "$env_dir/bin/python" -e "$checkout" --torch-backend=cu130
    uv pip install --python "$env_dir/bin/python" \
      'aiperf==0.12.0' 'flashinfer-python==0.6.18.post1' \
      'xgrammar==0.2.7' 'humming-kernels==0.1.15'
    ;;
  sglang)
    uv pip install --python "$env_dir/bin/python" --torch-backend=cu130 \
      --overrides <(printf '%s\n' 'flashinfer-python==0.6.18.post1' \
        'nvidia-cutlass-dsl[cu13]==4.7.1') \
      'sglang==0.5.17' 'sglang-kernel==0.4.5' \
      'torch==2.11.0' 'torchvision==0.26.0' \
      'aiperf==0.12.0' 'flash-attn-4==4.0.0b19'
    ;;
  *) echo 'Engine must be vllm or sglang' >&2; exit 2 ;;
esac
