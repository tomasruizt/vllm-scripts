#!/usr/bin/env bash
# Sync the port's files to the AMD host and launch one job there, in the background.
# Usage: bash remote.sh NAME SCRIPT.py ARGS... (runs in the ROCm image, log NAME.log)
# Env: HOST (g12-34), SSH_CONFIG, OVERLAY (default: the port and the CUDA stub).
set -euo pipefail
cd "$(dirname "$0")"
host=${HOST:-g12-34}
ssh_config=${SSH_CONFIG:-/private/tmp/claude-501/-Users-truiz-code-rh/f5fada5a-e3ba-4040-812d-54b1cb63ad3a/scratchpad/amd_ssh_config}
dir=runs/amd-port
name=$1
shift
ssh_() { ssh -F "$ssh_config" -o LogLevel=ERROR "$host" "$@"; }
ssh_ "mkdir -p $dir/benchmarks $dir/results"
scp -q -F "$ssh_config" -o LogLevel=ERROR topk_topp_triton_hier.py probe.py hist_bench.py sweep_gate.py phase_bench.py ../amd_bench.py debug_rounds.py gather_variants.py tier_bench.py ../gate.py ../large_k.py ../amd_run.sh "$host:$dir/"
overlay=${OVERLAY:-topk_topp_triton_hier.py topk_topp_cuda.py}
ssh_ "cd $dir && setsid nohup bash -c 'chg run --gpus 1 --timeout ${TIMEOUT:-30m} --note amd-port -- env OVERLAY=\"$overlay\" bash amd_run.sh $*; echo EXIT \$?' > $name.log 2>&1 < /dev/null &"
echo "launched $name on $host"
