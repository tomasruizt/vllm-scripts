#!/usr/bin/env bash
# One pass of the 112 report cases (8 to 2048 rows, V131k/V248k, real rows) for New Triton
# (triton_core.py) vs. Triton and CUDA, on a GPU host's PR tree under chg run. Async.
# Usage: triton_core_112.sh NAME  (log: ~/runs/triton-core-112/NAME/run.log on the host)
# Env: REMOTE, TREE.
set -euo pipefail
name=$1
remote=${REMOTE:-RH-nm-ore-b300-01}
tree=${TREE:-topk-cuda-pr2}
here=$(cd "$(dirname "$0")" && pwd)
opts=(-o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
out=runs/triton-core-112/$name
files=(triton_core.py triton_core_bench.py fastpath_bench.py fi_opt.py fi_fastpath.py large_k.py)

main() {
    local gpu
    gpu=$(mktemp -t tc112)
    gpu_script > "$gpu"
    ssh -T -n "${opts[@]}" "$remote" "mkdir -p $out code/topk-research"
    scp -q "${opts[@]}" "${files[@]/#/$here/}" "$remote:code/topk-research/"
    scp -q "${opts[@]}" "$gpu" "$remote:$out/gpu.sh"
    rm -f "$gpu"
    local run="chg run --gpus 1 --timeout 60m --note topk-triton-core-112 -- bash \$HOME/$out/gpu.sh"
    ssh -T -n "${opts[@]}" "$remote" "setsid nohup bash -c '$run; echo EXIT \$?' > $out/run.log 2>&1 < /dev/null & disown"
    echo "launched $remote:$out"
}

gpu_script() {
    cat <<EOF
set -u
cd \$HOME/code/$tree
export PATH=\$HOME/code/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:\$PATH
export PYTHONPATH=\$HOME/code/$tree:\$HOME/code/topk-research
python \$HOME/code/topk-research/triton_core_bench.py \$HOME/$out/pass-0.jsonl \\
  --logits-file \$HOME/code/qwen36-verify-logits-t1.pt --batch-sizes 8 16 64 128 256 512 1024 2048 \\
  --implementations triton-core triton cuda --fast-path "" > \$HOME/$out/bench.log 2>&1
echo GPU_DONE
EOF
}

main
