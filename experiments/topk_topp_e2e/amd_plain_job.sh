#!/usr/bin/env bash
# The PR head with the plain filter launch on MI355X: kernel tests and the 120-case report
# suite (exact, masks equal to the cooperative launch), then, if they pass, the E2E A/B as
# two chains on two GPU sets with swapped order. Run from the E2E run dir.
set -uo pipefail
run_dir=$PWD
(cd ~/runs/fallback-balance && chg run --gpus 1 --timeout 120m --note "topk plain-launch check" -- \
    env SUITES=report IMPLS="coop plain" PASSES=2 TRACE=0 TEST_KERNEL=plain_kernel.py OUT=plain- bash amd_gpu.sh)
bad=$(grep -c "False" ~/runs/fallback-balance/plain-report.log)
grep -E "passed|failed" ~/runs/fallback-balance/results/pytest.log | tail -n 1
if [[ $bad != 0 ]] || ! grep -E "passed|failed" ~/runs/fallback-balance/results/pytest.log | tail -n 1 | grep -q " passed" \
    || grep -E "passed|failed" ~/runs/fallback-balance/results/pytest.log | tail -n 1 | grep -q "failed"; then
    echo "CHECK_FAILED bad=$bad"; exit 1
fi
echo CHECK_OK
cd "$run_dir"
chg run --gpus 4 --timeout 150m --note "topk e2e A/B plain" -- env ORDER="main hier" bash amd_ab_full.sh 1 > full-a.log 2>&1 &
sleep 60
chg run --gpus 4 --timeout 150m --note "topk e2e A/B plain" -- env ORDER="hier main" bash amd_ab_full.sh 2 > full-b.log 2>&1 &
wait
echo JOB_DONE
