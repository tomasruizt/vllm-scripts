#!/usr/bin/env bash
# On the B200 host: debug_rounds traces for the reference (traced copy) and the port.
set -uo pipefail
export PATH=$HOME/code/topk-cuda-main/.venv/bin:$PATH TRITON_CACHE_DIR=$HOME/runs/amd-port/triton
cd ~/runs/amd-port
ln -sf ~/code/qwen36-verify-logits-t1.pt qwen36-verify-logits-t1.pt
for tree in ref port; do
  echo "TREE $tree"
  PYTHONPATH=$HOME/code/topk-triton-amd-$tree python debug_rounds.py --cases "$@" 2>&1 | grep "CASE\|TRACE\|EXACT"
done
