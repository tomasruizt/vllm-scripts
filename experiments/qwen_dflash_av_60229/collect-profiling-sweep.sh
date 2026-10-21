#!/usr/bin/env bash
# Local asynchronous collector. Profiling itself remains entirely in AIPerf.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
remote=${REMOTE:-RH-nm-ore-b300-01}
ssh -T "$remote" 'timeout 6300 bash -c '\''
    base=$HOME/runs/qwen-gdn-repeated-serving-20261020
    until test -f "$base/worker-a/exit-code.txt" && test -f "$base/worker-b/exit-code.txt"; do sleep 5; done
'\'''
make fetch-profiling-sweep REMOTE="$remote"
make analyze-profiling-sweep SWEEP_CHECK=--complete
echo DONE
