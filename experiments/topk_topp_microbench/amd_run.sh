#!/usr/bin/env bash
# Run a Python script from the current directory inside the ROCm vLLM image, on the
# GPUs chg reserved. Files in OVERLAY replace the image's vllm/v1/sample/ops modules.
# Usage, on the AMD host: chg run --gpus 1 -- bash amd_run.sh SCRIPT.py ARGS...
# Env: IMAGE, OVERLAY (e.g. "topk_topp_triton.py topk_topp_cuda.py").
set -euo pipefail
image=${IMAGE:-docker.io/vllm/vllm-openai-rocm:nightly-rocm100-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf}
ops=/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/ops
mounts=(-v "$PWD:/work:z")
for f in ${OVERLAY:-}; do mounts+=(-v "$PWD/$f:$ops/$f:ro,z"); done
mkdir -p tmp/triton
exec podman run --rm --device /dev/kfd --device /dev/dri --ipc=host \
  --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  -e TRITON_CACHE_DIR=/work/tmp/triton "${mounts[@]}" -w /work --entrypoint python3 "$image" "$@"
