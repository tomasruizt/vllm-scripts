#!/usr/bin/env bash
# CPU-only preparation. Run asynchronously before reserving GPUs.
set -euo pipefail
root="$HOME/runs/dsv41-cost-20261021"
envdir="$HOME/venvs/dsv41-cost-20261021"
commit=db3b0f0fae4bc98492b3863685e541e62bc59576
wheel="https://wheels.vllm.ai/$commit/vllm-0.32.1rc1.dev223%2Bgdb3b0f0fa-cp38-abi3-manylinux_2_28_x86_64.whl"
[[ -x "$envdir/bin/python" ]] || uv venv --python 3.12 "$envdir"
uv pip install --python "$envdir/bin/python" "$wheel" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
# Optional benchmark dependencies from this pinned main's bench extra.
uv pip install --python "$envdir/bin/python" pandas matplotlib seaborn datasets scipy plotly
export PATH="$envdir/bin:/usr/local/cuda-13.0/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.0
"$envdir/bin/vllm" download-kernels
"$envdir/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))' > "$root/preflight/installed-vllm-version.txt"
"$envdir/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' > "$root/preflight/site-packages.txt"
printf 'ENVIRONMENT_READY\n'
