#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p results
trap 'printf "%s\n" "$?" > results/finish-exit-code.txt' EXIT
ssh -T -o BatchMode=yes "${REMOTE:-RH-h100-3}" 'cd /data/tomasruizt/runs/capture-worst-case-20261020; timeout 7200 bash -c "until test -f dense/finished.txt && test -f gdn/finished.txt; do sleep 5; done"'
make fetch-results summarize
