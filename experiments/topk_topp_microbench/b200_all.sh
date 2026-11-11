#!/usr/bin/env bash
# B200 (SM100): build the PR tree, then tests with memcheck, CUDA vs. Triton (3 passes),
# and the FlashInfer fast path comparison. Usage, on the GPU host: b200_all.sh
set -euo pipefail
R=$HOME/code/topk-research
TREE=$HOME/code/topk-cuda-pr
TREE=$TREE bash "$R/pr_build.sh" 8af2080e71 10.0a "$HOME/code/pr66521-files.tgz"
bash "$R/tests_full.sh" "$TREE" "$HOME/code/tests-b200" memcheck
bash "$R/final_bench.sh" "$TREE" "$HOME/code/final6-b200"
bash "$R/fi_opt_bench.sh" "$TREE" "$HOME/code/fiopt6-b200" fi-fastpath fi-opt fi-opt-cub fi-opt-clusters fi-topk-only fill-only clone-only cuda triton
echo ALL_DONE
