#!/usr/bin/env bash
# Run a Triton core dev script on a GPU host (default B300) and wait for it.
# Usage: triton_core_dev.sh NAME SCRIPT.py ARGS...  (log: ~/runs/triton-core-dev/NAME.log)
# Env: REMOTE, TREE, RESEARCH (remote dir), EXTRA (more local files to upload), TRITON_ENV (e.g. TRITON_CACHE_DIR=...)
set -euo pipefail
name=$1
shift
remote=${REMOTE:-RH-nm-ore-b300-01}
here=$(cd "$(dirname "$0")" && pwd)
tree=${TREE:-topk-cuda-pr2}
research=${RESEARCH:-topk-research}
ssh_cmd=(ssh -T -n -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none "$remote")
[[ -n ${NO_UPLOAD:-} ]] || (cd "$here" && scp -q -o BatchMode=yes -o ControlMaster=no -o ControlPath=none triton_core*.py ${EXTRA:-} "$remote:code/$research/")
log="runs/triton-core-dev/$name.log"
env="export PATH=\$HOME/code/topk-cuda-main/.venv/bin:\$PATH PYTHONPATH=\$HOME/code/$tree:\$HOME/code/$research ${TRITON_ENV:-}"
cmd="cd ~/code/$tree; $env; chg run --gpus 1 --timeout ${GPU_TIMEOUT:-30m} --note triton-core-dev -- python \$HOME/code/$research/$*; echo EXIT \$?"
"${ssh_cmd[@]}" "mkdir -p runs/triton-core-dev; setsid nohup bash -c '$cmd' > $log 2>&1 < /dev/null & disown"
"${ssh_cmd[@]}" "timeout ${WAIT:-1200} bash -c 'until grep -q ^EXIT $log; do sleep 5; done'; tail -n ${TAIL:-30} $log"
