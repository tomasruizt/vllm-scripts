#!/usr/bin/env bash
# On the B200 host: the gate for the reference file, then for the port once per env
# setting (default: one run), in one GPU job.
# Usage: bash nv_gate.sh TAG ["HIER_X=1 HIER_Y=2" ...] (writes nv-TAG-ref.jsonl, nv-TAG-port{i}.jsonl)
set -uo pipefail
export PATH=$HOME/code/topk-cuda-main/.venv/bin:$PATH TRITON_CACHE_DIR=$HOME/runs/amd-port/triton
out=$HOME/runs/amd-port
tag=$1
shift
gate() {  # tree, output, env...
  local tree=$1 output=$2
  shift 2
  (cd ~/code/topk-triton-amd-$tree && env "$@" PYTHONPATH=$PWD:$HOME/code/topk-research python $out/gate.py \
    $output --logits-file ~/code/qwen36-verify-logits-t1.pt --impl triton-hier)
}
gate ref $out/nv-$tag-ref.jsonl HIER_NONE=1
i=0
for setting in "${@:-HIER_NONE=1}"; do
  gate port $out/nv-$tag-port$i.jsonl $setting
  i=$((i + 1))
done
