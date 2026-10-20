#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REMOTE=${REMOTE:-RH-nm-ore-b300-01}
RUN_ID=${RUN_ID:-qwen38-recipe-grid-20261019}
[[ $RUN_ID == qwen38-recipe-* ]]
mkdir -p "results/$RUN_ID"
trap 'printf "%s\n" "$?" > "results/$RUN_ID/postprocess-exit-code.txt"' EXIT
ssh -T -o BatchMode=yes "$REMOTE" bash -s -- "$RUN_ID" <<'REMOTE_WAIT'
set -euo pipefail
cd "$HOME/runs/$1"
timeout 10800 bash -c 'until test -f job-exit-code.txt; do sleep 5; done'
if ! grep -qx 0 job-exit-code.txt; then
    echo 'Remote recipe job failed:' >&2
    tail -n 15 driver.log >&2
    exit 1
fi
REMOTE_WAIT
make fetch-results REMOTE="$REMOTE" RUN_ID="$RUN_ID"
make summarize RUN_ID="$RUN_ID" CONCURRENCIES='1 8 32 64 128 256'
duckdb -c "SELECT CASE WHEN count(*)=54 AND bool_and(valid) AND count(DISTINCT (variant,concurrency,repeat))=54 THEN true ELSE error('Incomplete or invalid recipe comparison') END AS all_valid FROM read_parquet('results/$RUN_ID/metrics.parquet');"
make recipe-plot RUN_ID="$RUN_ID"
