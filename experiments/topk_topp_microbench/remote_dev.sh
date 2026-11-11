#!/usr/bin/env bash
# Launch dev_unified.py asynchronously on the B300 host.
# Usage: remote_dev.sh RUN_ID KERNEL_MODULE "dev_unified args..."
set -euo pipefail
run_id=$1 module=$2 args=$3
remote=${REMOTE:-RH-nm-ore-b300-01}
kernels=${KERNELS:-$HOME/code/rh/vllm-topk-topp-unified/benchmarks}
here=$(cd "$(dirname "$0")" && pwd)
ssh_cmd=(ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none "$remote")
"${ssh_cmd[@]}" "mkdir -p runs/$run_id"
scp -q -o BatchMode=yes "$here"/{dev_unified.py,benchmark_unified.py,benchmark_unified2.py,sweep_existing.py,ncu_unified.py} \
  "$kernels/kernels/$module.py" "$kernels/kernels/$module.cu" "$kernels/benchmark_topk_topp.py" \
  "$remote:runs/$run_id/"
venv=${REMOTE_VENV:-\$HOME/code/vllm/.venv}
"${ssh_cmd[@]}" "cd runs/$run_id && setsid nohup bash -c 'export PATH=$venv/bin:\$PATH; chg run --gpus 1 --timeout 40m --note topk-topp-microbench -- python dev_unified.py --output results --kernel-module $module $args; echo \$? > exit-code.txt' > launch.log 2>&1 < /dev/null &" < /dev/null
echo "launched $run_id"
