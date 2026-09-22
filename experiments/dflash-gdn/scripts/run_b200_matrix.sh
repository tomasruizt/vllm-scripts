#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=${1:?Specify output directory}
proposals=${2:-7}
parallel=${3:-7}
port=${4:-8500}
active=0
status=0
for model in 27B 4B 35B-A3B; do
    for variant in vllm_baseline vllm_dflash sglang_baseline sglang_dflash pr2_dflash; do
        if ((active >= parallel)); then
            wait -n || status=1
            active=$((active - 1))
        fi
        bash "$script_dir/run_b200.sh" "$model" "$output" "$port" "$proposals" "$variant" &
        active=$((active + 1))
        port=$((port + 1))
    done
done
while ((active > 0)); do
    wait -n || status=1
    active=$((active - 1))
done
exit "$status"
