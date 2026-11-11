#!/usr/bin/env bash
# MI355X, under chg: phase traces of MODS on LOGITS (made by make_synth.py if missing) at ROWS, REPEATS times.
set -u
export OVERLAY="topk_topp_cuda.py"
mkdir -p results/traces
[[ -f $LOGITS ]] || bash amd_run.sh make_synth.py $LOGITS 8 ${VOCAB:-248320} > synth.log 2>&1
for i in $(seq ${REPEATS:-3}); do
  for mod in $MODS; do
    bash amd_run.sh phase_trace.py $LOGITS --module=$mod ${FLAGS:-} $ROWS > results/traces/${TAG}_${mod}_$i.log 2>&1
    echo "trace $mod $i $?"
  done
done
if [[ -n ${TEST_KERNEL:-} ]]; then
  cp $TEST_KERNEL topk_topp_triton.py
  OVERLAY="topk_topp_triton.py topk_topp_sampler.py" bash amd_run.sh -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -k TritonTopkTopp -p no:cacheprovider > results/pytest_${TEST_KERNEL%.py}.log 2>&1
  echo "pytest $?"
fi
echo GPU_DONE
