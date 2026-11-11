#!/usr/bin/env bash
# One pass of the CuTe kernel and Triton (same process) over benchmark_topk_topp.py's configs and large k,
# on the GPU host from the CuTe tree. Usage: cute_onepass.sh OUT   (writes OUT/done.txt)
set -euo pipefail
OUT=$1
ROOT=$HOME/code
TREE=${TREE:-$ROOT/topk-cute}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$ROOT/topk-cuda-main/.venv/bin:$PATH PYTHONPATH=$TREE:$HERE
mkdir -p "$OUT"
cd "$TREE"
chg run --gpus 1 --timeout 60m --note topk-cute-onepass -- bash -c "
python $HERE/cute_hier_bench.py $OUT/pass-0.jsonl --logits-file $ROOT/qwen36-verify-logits-t1.pt --implementations triton cute > $OUT/pass-0.log 2>&1
python $HERE/cute_hier_bench.py $OUT/large-k-0.jsonl --logits-file $ROOT/qwen36-verify-logits-t1.pt --large-k --implementations triton cute > $OUT/large-k-0.log 2>&1" > "$OUT/run.log" 2>&1
echo $? > "$OUT/done.txt"
