#!/usr/bin/env bash
# Interleaved A/B of main vs the PR head (overlay-head/) over the full concurrency sweep, on
# the same 4 GPUs. ORDER sets which build goes first, so two chains on two GPU sets swap it.
# Usage, on the AMD host from the run dir: chg run --gpus 4 -- bash amd_ab_full.sh ROUND...
set -uo pipefail
export REPEATS=${REPEATS:-1 2 3} GSM8K_QUESTIONS=0 AIME_N=0
for round in "$@"; do
    for build in ${ORDER:-main hier}; do
        echo "SESSION $build-mtp5-full-r$round $(date -u +%FT%TZ)"
        OVERLAY=overlay-head bash amd_e2e.sh "$build-mtp5-full-r$round"
    done
done
echo AB_DONE
