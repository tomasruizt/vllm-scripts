#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT:-$HOME/runs/qwen38-capture-grid-20261019}"
trap 'printf "%s\n" "$?" > job-exit-code.txt' EXIT
trap 'exit 143' TERM INT
printf 'WAIT_DOWNLOAD %s\n' "$(date -u +%FT%TZ)"
timeout 1200 bash -c 'until test -f download-exit-code.txt; do sleep 2; done'
[[ $(<download-exit-code.txt) == 0 ]]
for variant in default 2k 4k; do
    grep -q 'Configuration valid' "configs/$variant-validation.txt"
done
printf 'RESERVE %s\n' "$(date -u +%FT%TZ)"
chg run --gpu-ids 0,1,2,3 --wait 5m --timeout 45m --note 'Qwen3.8 BF16 capture grid Default / Up to 2K / Up to 4K' -- bash run.sh
