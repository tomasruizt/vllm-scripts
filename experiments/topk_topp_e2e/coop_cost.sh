#!/usr/bin/env bash
# In-container entry for amd_e2e.sh: RUN=coop_cost.sh bash amd_e2e.sh hier-x ROWS.pt
exec python3 coop_cost.py "$@"
