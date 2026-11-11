#!/usr/bin/env bash
# Interleaved A/B of main vs the PR with a plain (non-cooperative) filter launch on ROCm
# (overlay-plain/), on the same 4 GPUs. Usage: chg run --gpu-ids A,B,C,D -- bash amd_ab_plain.sh
set -uo pipefail
export REPEATS=${REPEATS:-1 2} CONCURRENCIES=${CONCURRENCIES:-1 8} GSM8K_QUESTIONS=0 AIME_N=0
rm -rf overlay-plain && cp -r overlay overlay-plain
sed -i 's/^FILTER_OPTS = dict(num_warps=WARPS, launch_cooperative_grid=True)$/FILTER_OPTS = dict(num_warps=WARPS, launch_cooperative_grid=not current_platform.is_rocm())/' \
    overlay-plain/vllm/v1/sample/ops/topk_topp_triton.py
grep -n "^FILTER_OPTS" overlay-plain/vllm/v1/sample/ops/topk_topp_triton.py
for round in 1 2; do
    echo "SESSION main $(date -u +%FT%TZ)"; bash amd_e2e.sh "main-mtp5-abp$round"
    echo "SESSION hier-plain $(date -u +%FT%TZ)"; OVERLAY=overlay-plain bash amd_e2e.sh "hier-mtp5-plain-abp$round"
done
echo AB_DONE
