#!/usr/bin/env bash
# One-time setup of an AMD run dir inside the ROCm image: python shim and AIPerf venv.
set -euo pipefail
image=${IMAGE:-docker.io/vllm/vllm-openai-rocm:nightly-rocm100-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf}
mkdir -p home py/bin
podman run --rm -v "$PWD:/work:z" -w /work --entrypoint bash "$image" -c '
ln -sf "$(command -v python3)" py/bin/python
python3 -m venv aiperf-venv && aiperf-venv/bin/pip install -q aiperf==0.13.0
aiperf-venv/bin/aiperf --version; command -v curl setsid timeout'
