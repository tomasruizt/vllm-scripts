#!/usr/bin/env bash
# Wait until two GPUs are free, then run the AV on and off nsys profiles on them. Usage: queue-nsys.sh <out_dir>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$1

free_gpus() {
    canhazgpu status 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | awk '/AVAILABLE/ {print $1}'
}

until [ "$(free_gpus | wc -l)" -ge 2 ]; do sleep 60; done
read -r gpu_on gpu_off <<< "$(free_gpus | head -2 | tr '\n' ' ')"
echo "launching on GPUs $gpu_on (AV on) and $gpu_off (AV off) at $(date -u +%FT%TZ)"
bash "$here/run-nsys.sh" "$gpu_on" 8650 on "$out/on" &
bash "$here/run-nsys.sh" "$gpu_off" 8651 off "$out/off" &
wait
echo QUEUE-DONE
