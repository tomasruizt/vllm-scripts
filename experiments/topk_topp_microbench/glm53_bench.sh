#!/usr/bin/env bash
# One pass of real_topp on the GLM-5.3-Flash verify rows (V154880), main's Triton vs. the
# group-maxima kernel (`new`), plus the clone-only time, on a GPU host's tree under chg run. Async.
# Usage: glm53_bench.sh NAME  (log: ~/runs/glm53-bench/NAME/run.log on the host)
# Env: REMOTE, TREE.
set -euo pipefail
name=$1
remote=${REMOTE:-RH-nm-ore-b300-01}
tree=${TREE:-topk-cuda-pr2}
here=$(cd "$(dirname "$0")" && pwd)
opts=(-o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
out=runs/glm53-bench/$name
research=code/topk-glm53

main() {
    ssh -T -n "${opts[@]}" "$remote" "mkdir -p $out $research"
    scp -q "${opts[@]}" "$here/glm53_bench.py" "$here/clone_only.py" \
        "$HOME/code/rh/vllm-topk-topp-unified/benchmarks/benchmark_topk_topp.py" "$remote:$research/"
    scp -q "${opts[@]}" "$HOME/code/rh/vllm-topk-triton-hier/vllm/v1/sample/ops/topk_topp_triton.py" \
        "$remote:$research/topk_topp_triton_new.py"
    gpu_script | ssh -T "${opts[@]}" "$remote" "cat > $out/gpu.sh"
    local run="chg run --gpus 1 --wait 60m --timeout 30m --note glm53-topp-bench -- bash \$HOME/$out/gpu.sh"
    ssh -T -n "${opts[@]}" "$remote" "setsid nohup bash -c '$run; echo EXIT \$?' > $out/run.log 2>&1 < /dev/null & disown"
    echo "launched $remote:$out"
}

gpu_script() {
    cat <<EOF
set -u
cd \$HOME/code/$tree
out=\$HOME/$out
export PATH=\$HOME/code/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:\$PATH
export PYTHONPATH=\$HOME/code/$tree:\$HOME/$research
nvidia-smi --query-gpu=name --format=csv,noheader -i \$CUDA_VISIBLE_DEVICES > \$out/gpu.txt
python \$HOME/$research/glm53_bench.py --batch-sizes 8 16 64 128 256 512 1024 2048 \\
  --vocab-sizes 154880 --logits-file \$HOME/code/glm53-verify-logits.pt --scenarios real_topp \\
  --implementations triton new --check --output \$out/pass-0.jsonl > \$out/pass-0.log 2>&1
echo "bench \$?"
python \$HOME/$research/clone_only.py \$out/clone.jsonl 154880 > \$out/clone.log 2>&1
echo "clone \$?"
echo GPU_DONE
EOF
}

main
