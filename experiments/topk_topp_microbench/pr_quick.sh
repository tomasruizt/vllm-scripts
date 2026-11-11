#!/usr/bin/env bash
# Incremental rebuild of the PR tree after copying in a new kernel source, then
# the exact-filter tests and per-kernel times. Usage: pr_quick.sh TREE CU CASES...
set -euo pipefail
TREE=$1 CU=$2
shift 2
export PATH=$HOME/code/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:$PATH PYTHONPATH=$TREE
cd "$TREE"
cp "$CU" csrc/topk_topp_filter.cu
cmake --build cmake-build-release --target _C_stable_libtorch -j 64 > quick-build.log 2>&1 || { tail -n 30 quick-build.log; exit 1; }
cmake --install cmake-build-release --component _C_stable_libtorch >> quick-build.log
cuobjdump --dump-resource-usage vllm/_C_stable_libtorch.abi3.so 2>/dev/null | grep -A1 topk_topp_filter | grep -oE "REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+" | paste -sd' '
python -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -p no:cacheprovider -k "Cuda" 2>&1 | tail -n 1
python kernel_times.py "$@" 2>/dev/null | grep "us$"
