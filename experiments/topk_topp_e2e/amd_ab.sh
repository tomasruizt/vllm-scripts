#!/usr/bin/env bash
# Interleaved A/B of main vs the PR on the same 4 GPUs (one chg reservation), at low concurrency only.
# Usage, on the AMD host from the run dir: chg run --gpu-ids A,B,C,D -- bash amd_ab.sh [ROUNDS]
set -uo pipefail
rounds=${1:-2}
export REPEATS=${REPEATS:-1 2} CONCURRENCIES=${CONCURRENCIES:-1 8} GSM8K_QUESTIONS=0 AIME_N=0
for round in $(seq 1 "$rounds"); do
    for build in main hier; do
        echo "SESSION $build-mtp5-ab$round $(date -u +%FT%TZ)"
        bash amd_e2e.sh "$build-mtp5-ab$round"
    done
done
echo AB_DONE
