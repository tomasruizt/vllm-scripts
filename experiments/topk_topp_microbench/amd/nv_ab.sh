#!/usr/bin/env bash
# On the B200 host: the gate for several trees (~/code/topk-triton-amd-TREE), in one GPU job.
# Usage: bash nv_ab.sh TAG TREE... (writes ~/runs/amd-port/nv-TAG-TREE.jsonl)
set -uo pipefail
export PATH=$HOME/code/topk-cuda-main/.venv/bin:$PATH TRITON_CACHE_DIR=$HOME/runs/amd-port/triton
out=$HOME/runs/amd-port
tag=$1
shift
i=0
for tree in "$@"; do
  i=$((i + 1))
  (cd ~/code/topk-triton-amd-$tree && PYTHONPATH=$PWD:$HOME/code/topk-research python $out/gate.py \
    $out/nv-$tag-$i-$tree.jsonl --logits-file ~/code/qwen36-verify-logits-t1.pt --impl triton-hier)
done
