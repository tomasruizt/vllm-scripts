#!/usr/bin/env bash
# Two chains on two GPUs, each serving every variant in turn, in opposite order,
# so every variant runs on both GPUs and no variant is pinned to one GPU or one time slot.
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-13.0
export PATH="$HOME/code/av-sync/main/.venv/bin:$CUDA_HOME/bin:$PATH"
VARIANTS=${VARIANTS:-k5 av_main av_depth3 av_ring3 av_lazy}
REVERSED=$(tr ' ' '\n' <<< "$VARIANTS" | tac | tr '\n' ' ')
GPUS=${GPUS_PER_CHAIN:-1} PORT_BASE=${PORT_BASE:-8860}
PORT=$PORT_BASE chg run --gpus "$GPUS" --timeout 3h --note "AV confidence sync A/B (chain 1)" -- python3 run.py $VARIANTS > chain1.log 2>&1 &
PID1=$!
PORT=$((PORT_BASE + 1)) chg run --gpus "$GPUS" --timeout 3h --note "AV confidence sync A/B (chain 2)" -- python3 run.py $REVERSED > chain2.log 2>&1 &
wait $PID1; R1=$?
wait $!; exit $((R1 | $?))
