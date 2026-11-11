#!/usr/bin/env bash
# coop_probe.py in fresh processes, alternating modes. Usage: chg run --gpus 1 -- bash amd_probe.sh
for round in 1; do
    for mode in none coop coop-sync plain; do RUN=coop_probe.sh bash amd_e2e.sh hier-probe /work/glm-rows.pt $mode 2>&1 | grep "mode=\|Error"; done
done
echo PROBE_DONE
