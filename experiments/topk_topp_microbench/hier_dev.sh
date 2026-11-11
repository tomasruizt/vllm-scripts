#!/usr/bin/env bash
# Build the hierarchical CUDA filter on a GPU host (default B300, without a GPU), then run
# a script there under chg and wait for it. Copies hier_* and the helpers it imports
# into ~/code/topk-research/hier (first on PYTHONPATH).
# Usage: hier_dev.sh NAME SCRIPT.py ARGS...  (log: ~/runs/hier-dev/NAME.log)
# Env: REMOTE, TREE, ARCH (TORCH_CUDA_ARCH_LIST), GPU_TIMEOUT, WAIT, TAIL, NOWAIT=1.
set -euo pipefail
name=$1
shift
remote=${REMOTE:-RH-nm-ore-b300-01}
here=$(cd "$(dirname "$0")" && pwd)
opts=(-o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none)
ssh_cmd=(ssh -T -n "${opts[@]}" "$remote")
dir=code/topk-research/hier
"${ssh_cmd[@]}" "mkdir -p $dir runs/hier-dev"
scp -qp "${opts[@]}" "$here"/hier_*.py "$here"/hier_cuda.cu "$here"/{large_k,fastpath_bench,fi_opt,fi_fastpath,triton_core,triton_core_bench}.py \
  "$remote:$dir/"
log="runs/hier-dev/$name.log"
tree=${TREE:-topk-cuda-pr2}
env="export PATH=\$HOME/code/topk-cuda-main/.venv/bin:\$PATH PYTHONPATH=\$HOME/$dir:\$HOME/code/$tree"
env+=" TORCH_CUDA_ARCH_LIST=${ARCH:-10.3} MAX_JOBS=4"
build="CUDA_VISIBLE_DEVICES= nice python \$HOME/$dir/hier_cuda.py"
run="chg run --gpus 1 --timeout ${GPU_TIMEOUT:-30m} --note hier-dev -- python \$HOME/$dir/$*"
cmd="cd ~/code/$tree; $env; $build && $run; echo EXIT \$?"
"${ssh_cmd[@]}" "setsid nohup bash -c '$cmd' > $log 2>&1 < /dev/null & disown"
[[ -n ${NOWAIT:-} ]] && exit 0
"${ssh_cmd[@]}" "timeout ${WAIT:-1200} bash -c 'until grep -q ^EXIT $log; do sleep 5; done'; tail -n ${TAIL:-30} $log"
