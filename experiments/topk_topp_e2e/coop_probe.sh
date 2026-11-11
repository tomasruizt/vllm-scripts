#!/usr/bin/env bash
# In-container entry for amd_e2e.sh: RUN=coop_probe.sh bash amd_e2e.sh hier-x ROWS.pt MODE
exec python3 coop_probe.py "$@"
