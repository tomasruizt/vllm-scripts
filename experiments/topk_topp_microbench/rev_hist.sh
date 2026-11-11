#!/usr/bin/env bash
# rev_hist.py on a GPU host under chg run, async. Kernel copies from results/rev-hist/src.
# Usage: rev_hist.sh launch NAME ARGS... | wait NAME | fetch NAME   (ARGS go to rev_hist.py)
# Env: REMOTE (default B200), TREE (vLLM tree for the imports), PROCS (processes, odd ones --swap).
# Remote dir: ~/runs/rev-hist/NAME.
set -euo pipefail
cmd=$1 name=$2
shift 2
remote=${REMOTE:-RH-nm-ohio-b200-01}
tree=${TREE:-topk-cuda-pr}
here=$(cd "$(dirname "$0")" && pwd)
opts=(-o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
out=runs/rev-hist/$name
files=("$here"/{rev_hist,gate,large_k,nv_after_rocm}.py "$here"/bench_fixed/benchmark_topk_topp.py "$here"/results/rev-hist/src/*.py)

launch() {
    ssh -T -n "${opts[@]}" "$remote" "mkdir -p $out/code"
    scp -q "${opts[@]}" "${files[@]}" "$remote:$out/code/"
    gpu_script "$@" | ssh -T "${opts[@]}" "$remote" "cat > $out/gpu.sh"
    local run="chg run --gpus 1 --wait 30m --timeout 60m --note topk-rev-hist -- bash \$HOME/$out/gpu.sh"
    ssh -T -n "${opts[@]}" "$remote" "setsid nohup bash -c '$run; echo EXIT \$?' > $out/run.log 2>&1 < /dev/null & disown"
    echo "launched $remote:$out"
}

gpu_script() {
    cat <<EOF
set -u
cd \$HOME/$out/code
export PATH=\$HOME/code/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:\$PATH
export PYTHONPATH=\$HOME/code/$tree:\$HOME/$out/code TRITON_CACHE_DIR=\$HOME/$out/triton
for i in \$(seq 0 $((${PROCS:-1} - 1))); do  # odd processes load rev first
    swap=\$([ \$((i % 2)) = 1 ] && echo --swap)
    python rev_hist.py \$HOME/$out/results --qwen-rows \$HOME/code/qwen36-verify-logits-t1.pt --glm-rows \$HOME/code/glm53-verify-logits.pt --proc \$i \$swap $* >> \$HOME/$out/bench.log 2>&1
    echo "bench \$i \$?"
done
EOF
}

case $cmd in
launch) launch "$@" ;;
wait) ssh -T -n "${opts[@]}" "$remote" "timeout ${TIMEOUT:-3000} bash -c 'until grep -q EXIT $out/run.log; do sleep 10; done'; tail -n 3 $out/run.log; tail -n 3 $out/bench.log" ;;
fetch) mkdir -p "$here/results/rev-hist/$name" && scp -q "${opts[@]}" "$remote:$out/results/*.jsonl" "$here/results/rev-hist/$name/" ;;
esac
