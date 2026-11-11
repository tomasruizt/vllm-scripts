#!/usr/bin/env bash
# MI355X, under chg: SUITES (gate tiles report) of balance.py, the phase traces, and the
# kernel tests with the fix as topk_topp_triton.py. Run from the run dir.
set -u
logits=(--qwen-rows qwen.pt --glm-rows glm.pt)
for suite in ${SUITES:-gate tiles}; do
  OVERLAY="topk_topp_cuda.py" bash amd_run.sh balance.py results/${OUT:-}$suite "${logits[@]}" --suite $suite --passes ${PASSES:-2} --impls ${IMPLS:-pr fix} > ${OUT:-}$suite.log 2>&1
  echo "$suite $?"
done
if [[ ${TRACE:-1} == 1 ]]; then
  for mod in trace_pr_kernel trace_fix_kernel; do
    for shuf in "" --shuffle; do
      log=results/traces/${mod}${shuf:+_shuffled}.log
      mkdir -p results/traces
      OVERLAY="topk_topp_cuda.py" bash amd_run.sh phase_trace.py qwen.pt --module=$mod $shuf 512 1024 2048 > $log 2>&1
      echo "trace $mod$shuf $?"
    done
  done
fi
if [[ ${TESTS:-1} == 1 ]]; then
  cp ${TEST_KERNEL:-fix_kernel.py} topk_topp_triton.py
  OVERLAY="topk_topp_triton.py topk_topp_sampler.py" bash amd_run.sh -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -k TritonTopkTopp -p no:cacheprovider > results/pytest.log 2>&1
  echo "pytest $?"
fi
echo GPU_DONE
