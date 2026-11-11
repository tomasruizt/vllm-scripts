#!/usr/bin/env bash
# In-container entry for amd_e2e.sh: RUN=host_overhead.sh bash amd_e2e.sh {main,hier}-x [ROWS.pt]
exec python3 host_overhead.py "$@"
