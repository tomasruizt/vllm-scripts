#!/usr/bin/env bash
# Run only inside a four-GPU canhazgpu reservation. A failed smoke test stops here.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${DSV41_ROOT:?Set the new full-run directory}
model=/data/engine/hub_cache/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
clone=/data/tomasruizt/hub_cache/models--inference-optimization--DeepSeek-V4-Flash-DSpark-2.1B-A1.9B-small/snapshots/bbddd703cacc556881da865ecb67663454e943ab
trap 'echo "$?" > "$root/pipeline-exit-code.txt"' EXIT
echo SMOKE_START
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" DSV41_SMOKE=1 DSV41_ROOT="$root/smoke" \
    DSV41_MODEL="$clone" DSV41_TOKENIZER="$model" DSV41_PORT=8682 bash "$here/run-dsv41-cost.sh"
test -f "$root/smoke/DONE"
echo SMOKE_PASSED
env DSV41_CALIBRATION_SOURCE="$HOME/runs/dsv41-k5-full-20261023-r1" bash "$here/run-dsv41-cost.sh"
