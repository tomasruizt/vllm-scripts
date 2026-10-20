#!/usr/bin/env bash
# Launch asynchronously from SSH. One persistent TP2 server per cost-data fold.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$1
gpus=$2
port=$3
seed=$4
mkdir -p "$root"
cp "$0" "$root/launch.sh"
printf 'OMP_NUM_THREADS=1\nTENSOR_PARALLEL_SIZE=2\nMAX_NUM_SEQS=256\n' > "$root/launch-env.txt"
exec canhazgpu run --gpu-ids "$gpus" --timeout 50m --note "Qwen AV c256 TP2 measured costs rh-fll" -- \
    env ROOT="$root" OMP_NUM_THREADS=1 MAX_NUM_SEQS=256 TENSOR_PARALLEL_SIZE=2 \
    VLLM_CHECKOUT="${VLLM_CHECKOUT:-$HOME/code/vllm-cost-c256-tp2-60229}" AV_COST_SEED="$seed" \
    AV_COST_LOOKUP="$here/cost-lookup-frozen.json" AV_COST_PLAN="${AV_COST_PLAN:-$here/cost-c256-collection-plan.json}" \
    bash "$here/run-cost-oracle.sh" "${gpus%%,*}" "$port"
