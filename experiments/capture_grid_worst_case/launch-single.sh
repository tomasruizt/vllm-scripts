#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data/tomasruizt/runs/capture-worst-case-20261020}
cd "$ROOT"
CASE=${1:?Specify dense or gdn}
source environment.sh
trap 'printf "%s\n" "$?" > "$CASE-launch-exit-code.txt"' EXIT
chg run --gpus 1 --wait 5m --timeout 75m --note "H100 capture-grid $CASE worst-case pilot" -- "$EXPERIMENT_PYTHON" run.py "$CASE"
"$EXPERIMENT_PYTHON" collect.py "$ROOT"
