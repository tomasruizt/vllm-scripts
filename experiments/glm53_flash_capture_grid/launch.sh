#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT:-$HOME/runs/glm53-flash-capture-pilot-20261019}"
export ROOT=$PWD
export HELPERS=$PWD/common.sh
trap 'printf "%s\n" "$?" > job-exit-code.txt' EXIT
trap 'exit 143' TERM INT
chg run --gpu-ids 0,1,2,3 --wait 30m --timeout 40m --note 'GLM5.3 Flash Default vs Up to4K pilot C64/C128' -- bash run.sh
