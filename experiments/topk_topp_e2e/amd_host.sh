#!/usr/bin/env bash
# Host cost and implicit-sync check of the filter call, main then the PR, on one GPU.
# Usage, on the AMD host from the run dir: chg run --gpus 1 -- bash amd_host.sh [ROWS.pt]
for build in main hier; do RUN=host_overhead.sh bash amd_e2e.sh "$build-host" "$@"; done
echo HOST_DONE
