#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-$HOME/runs/qwen38-recipe-restarts-20261020}
cd "$BASE"
export BASE
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
chg run --gpu-ids 0,1,2,3 --wait 30m --timeout 60m --note 'Qwen capture startup repeats, worker0' -- bash restart-run.sh 0 > worker-0.log 2>&1 &
FIRST_PID=$!
chg run --gpu-ids 4,5,6,7 --wait 30m --timeout 60m --note 'Qwen capture startup repeats, worker1' -- bash restart-run.sh 1 > worker-1.log 2>&1 &
SECOND_PID=$!
trap 'kill -TERM "$FIRST_PID" "$SECOND_PID" 2>/dev/null || true; exit 143' TERM INT
RESULT=0
wait "$FIRST_PID" || RESULT=1
wait "$SECOND_PID" || RESULT=1
exit "$RESULT"
