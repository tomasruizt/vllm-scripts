#!/usr/bin/env bash
# Push the CuTe tree to the GPU host, run the fast gate (cute_hier_gate.py) on one GPU, print the verdict.
# Usage: cute_hier_gate.sh RUN [BEST_JSONL]   (results in results/cute-hier-dev/RUN.jsonl)
set -euo pipefail
run=$1 best=${2:-}
host=${HOST:-RH-nm-ore-b300-01}
here=$(cd "$(dirname "$0")" && pwd)
dir=runs/cute-dev/$run
ssh_=(ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none "$host")
bash "$here/cute_push.sh" "$host" '$HOME/code/topk-cute-dev' | tail -1
"${ssh_[@]}" "mkdir -p $dir"
scp -q -o BatchMode=yes "$here/cute_hier_gate.py" "$here/large_k.py" ${best:+"$best"} "$host:$dir/"
best_arg=${best:+--best \$HOME/$dir/$(basename "$best")}
"${ssh_[@]}" "cd \$HOME/code/topk-cute && export PATH=\$HOME/code/topk-cuda-main/.venv/bin:\$PATH PYTHONPATH=\$HOME/code/topk-cute:\$HOME/$dir; GATE_IMPL=${GATE_IMPL:-cute-hier} timeout 900 chg run --gpus 1 --timeout 15m --note topk-cute-gate -- python \$HOME/$dir/cute_hier_gate.py \$HOME/$dir/out.jsonl --logits-file \$HOME/code/qwen36-verify-logits-t1.pt $best_arg > \$HOME/$dir/log.txt 2>&1; grep -E 'EDGE|^edges|µs|/[0-9]+/[0-9]+ |^GATE|Error' \$HOME/$dir/log.txt | tail -30"
mkdir -p "$here/results/cute-hier-dev"
scp -q -o BatchMode=yes "$host:$dir/out.jsonl" "$here/results/cute-hier-dev/$run.jsonl"
