#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT:-$HOME/runs/qwen38-recipe-grid-20261019}"
trap 'printf "%s\n" "$?" > job-exit-code.txt' EXIT
trap 'exit 143' TERM INT
printf 'WAIT_DOWNLOAD %s\n' "$(date -u +%FT%TZ)"
timeout 7200 bash -c 'until test -f download-exit-code.txt; do sleep 5; done'
[[ $(<download-exit-code.txt) == 0 ]]
[[ $(find configs -name '*-validation.txt' | wc -l) == 54 ]]
! grep -L 'Configuration valid' configs/*/*-validation.txt | grep .
printf 'RESERVE %s\n' "$(date -u +%FT%TZ)"
chg run --gpu-ids 0,1,2,3 --wait 30m --timeout 70m --note 'Qwen3.8 FP8 recipe capture grid, 3 repeats' -- bash recipe-run.sh
