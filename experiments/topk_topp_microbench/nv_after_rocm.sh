#!/usr/bin/env bash
# nv_after_rocm.py on a GPU host's tree (main's Triton + the unified benchmark) under chg run. Async.
# Usage: nv_after_rocm.sh NAME ARGS...  (ARGS go to nv_after_rocm.py, log: ~/runs/nv-after-rocm/NAME/run.log)
# Env: REMOTE, TREE, RUN_ROOT (runs/ subdir), EXTRA_FILES (more files for code/). Kernel copies come from results/nv-after-rocm/src.
set -euo pipefail
name=$1
shift
remote=${REMOTE:-RH-nm-ore-b300-01}
tree=${TREE:-topk-cuda-pr2}
here=$(cd "$(dirname "$0")" && pwd)
opts=(-o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
out=runs/${RUN_ROOT:-nv-after-rocm}/$name
files=("$here"/{nv_after_rocm,gate,large_k}.py "$here"/results/nv-after-rocm/src/*_kernel.py ${EXTRA_FILES:-})

main() {
    ssh -T -n "${opts[@]}" "$remote" "mkdir -p $out/code"
    scp -q "${opts[@]}" "${files[@]}" "$remote:$out/code/"
    gpu_script "$@" | ssh -T "${opts[@]}" "$remote" "cat > $out/gpu.sh"
    local run="chg run --gpus 1 --wait 30m --timeout 45m --note topk-nv-after-rocm -- bash \$HOME/$out/gpu.sh"
    ssh -T -n "${opts[@]}" "$remote" "setsid nohup bash -c '$run; echo EXIT \$?' > $out/run.log 2>&1 < /dev/null & disown"
    echo "launched $remote:$out"
}

gpu_script() {
    cat <<EOF
set -u
cd \$HOME/code/$tree
export PATH=\$HOME/code/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:\$PATH
export PYTHONPATH=\$HOME/code/$tree:\$HOME/$out/code TRITON_CACHE_DIR=\$HOME/$out/triton
python \$HOME/$out/code/nv_after_rocm.py \$HOME/$out/results $* > \$HOME/$out/bench.log 2>&1
echo "bench \$?"
EOF
}

main "$@"
