#!/usr/bin/env bash
# Forced split-V sweep at TP1 and TP2 shapes on the tuning build. Run from the run directory inside chg.
set -euo pipefail
CODE=${CODE:-$HOME/code}
MAIN=${MAIN:-$CODE/gdn-k8-verify-b300-main}
export PATH=$MAIN/.venv/bin:$PATH PYTHONPATH=$CODE/gdn-k8-splitv-tune
for tp in 1 2; do
    python kernel_bench.py --label tune --tp $tp --impls cuda --splits auto 1 2 4 \
        --batches 1 2 3 4 6 8 12 16 24 32 > "sweep-tp$tp.jsonl"
done
echo SWEEP_DONE
