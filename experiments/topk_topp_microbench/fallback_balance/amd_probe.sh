#!/usr/bin/env bash
# MI355X, under chg: phase traces on synthetic Zipf rows, and IMPLS on a subset (ONLY) of the gate.
set -u
export OVERLAY="topk_topp_cuda.py"
mkdir -p results/traces
[[ -f synth8.pt ]] || bash amd_run.sh make_synth.py synth8.pt 8 248320 > synth.log 2>&1
for mod in ${TRACES-trace_pr_kernel trace_fix_kernel}; do
  bash amd_run.sh phase_trace.py synth8.pt --module=$mod 8 256 > results/traces/synth_$mod.log 2>&1
  echo "trace $mod $?"
done
bash amd_run.sh balance.py results/${OUT:-probe} --qwen-rows qwen.pt --suite gate --passes 2 --impls $IMPLS --only $ONLY > probe.log 2>&1
echo "probe $?"
[[ -n ${TILES:-} ]] && bash amd_run.sh balance.py results/${OUT:-probe}-tiles --qwen-rows qwen.pt --suite tiles --passes 2 --impls $IMPLS > probe_tiles.log 2>&1 && echo "tiles $?"
for shuf in ${REAL_TRACES:-}; do
  bash amd_run.sh phase_trace.py qwen.pt --module=trace_fix3_kernel ${shuf/tiled/} 512 1024 2048 > results/traces/real_fix3_$shuf.log 2>&1
done
echo GPU_DONE
