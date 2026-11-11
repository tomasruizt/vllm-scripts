#!/usr/bin/env bash
# NVIDIA, under chg: SUITES of balance.py, then the kernel tests on tree/ (the PR's sampler
# and tests, the fix as topk_topp_triton.py). Run from the run dir. Env: VENV, QWEN, GLM.
set -u
run=$PWD
export PATH=${VENV:-$HOME/code/topk-cuda-main/.venv}/bin:/usr/local/cuda-13.0/bin:$PATH
export PYTHONPATH=$run/tree:$run TRITON_CACHE_DIR=$run/triton
mkdir -p results
nvidia-smi --query-gpu=name --format=csv,noheader > results/gpu.txt
for suite in ${SUITES:-gate tiles}; do
  python balance.py results/${OUT:-}$suite --qwen-rows $QWEN --glm-rows $GLM --suite $suite --passes ${PASSES:-2} --impls ${IMPLS:-pr fix} > ${OUT:-}$suite.log 2>&1
  echo "$suite $?"
done
if [[ ${TESTS:-1} == 1 ]]; then
  (cd tree && python -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -k TritonTopkTopp -p no:cacheprovider > $run/results/pytest.log 2>&1)
  echo "pytest $?"
fi
echo GPU_DONE
