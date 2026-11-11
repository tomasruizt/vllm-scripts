#!/usr/bin/env bash
# Sync the PR files of the local worktree into a GPU host's PR tree, rebuild _C_stable_libtorch,
# then under one chg reservation: the sampler tests (and memcheck), 3 passes of Triton vs. CUDA,
# large k, and (RELATED=1) the related-PR arms. Async: returns after launch.
# Usage: pr_all.sh NAME  (log: ~/runs/pr-all/NAME/run.log on the host)
# Env: REMOTE, TREE, LOCAL, BENCH=1 (one benchmark pass only), QUICK=1 (tests of the file only), MEMCHECK=1, RELATED=1, GPU_TIMEOUT.
set -euo pipefail
name=$1
remote=${REMOTE:-RH-nm-ore-b300-01}
tree=${TREE:-topk-cuda-pr2}
local_tree=${LOCAL:-$HOME/code/rh/vllm-topk-topp-unified}
here=$(cd "$(dirname "$0")" && pwd)
opts=(-o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none)
ssh_cmd=(ssh -T -n "${opts[@]}" "$remote")
out=runs/pr-all/$name

main() {
    sync_files
    "${ssh_cmd[@]}" "setsid nohup bash -c '$(remote_cmd)' > $out/run.log 2>&1 < /dev/null & disown"
    echo "launched $remote:$out"
}

# The PR's files plus those it no longer touches, so the remote tree matches the worktree.
sync_files() {
    local tarball files
    tarball=$(mktemp -t pr-all).tgz
    files=$(cd "$local_tree" && { git diff --name-only 8af2080e71; git diff --name-only 8af2080e71 72544ce7f5; } | sort -u)
    (cd "$local_tree" && COPYFILE_DISABLE=1 tar --no-xattrs -czf "$tarball" $files)
    "${ssh_cmd[@]}" "mkdir -p $out code/topk-research"
    scp -q "${opts[@]}" "$tarball" "$remote:$out/files.tgz"
    scp -q "${opts[@]}" "$here"/{large_k.py,related_prs.sh,fastpath_bench.py,fi_fastpath.py,fi_opt.py} "$remote:code/topk-research/"
    gpu_script > "$tarball.gpu.sh"
    scp -q "${opts[@]}" "$tarball.gpu.sh" "$remote:$out/gpu.sh"
    rm -f "$tarball" "$tarball.gpu.sh"
}

remote_cmd() {
    local env="export PATH=\$HOME/code/topk-cuda-main/.venv/bin:/usr/local/cuda-13.0/bin:\$PATH PYTHONPATH=\$HOME/code/$tree"
    local build="tar -xzf \$HOME/$out/files.tgz && cmake --build cmake-build-release --target _C_stable_libtorch -j 32 > \$HOME/$out/build.log 2>&1 && cmake --install cmake-build-release --component _C_stable_libtorch >> \$HOME/$out/build.log"
    local gpu="chg run --gpus 1 --timeout ${GPU_TIMEOUT:-300m} --note topk-pr-all -- bash \$HOME/$out/gpu.sh"
    echo "cd ~/code/$tree; $env; $build && echo BUILD_OK && $gpu; echo EXIT \$?"
}

gpu_script() {
    cat <<EOF
set -u
cd \$HOME/code/$tree
out=\$HOME/$out
f=tests/v1/sample/test_topk_topp_sampler.py
if [ -n "${BENCH:-}" ]; then
  python benchmarks/benchmark_topk_topp.py --batch-sizes 8 16 64 128 256 512 1024 2048 \
    --vocab-sizes 131072 248320 --logits-file \$HOME/code/qwen36-verify-logits-t1.pt \
    --implementations triton cuda --check --output \$out/pass-0.jsonl > \$out/pass-0.log 2>&1
  echo GPU_DONE; exit 0
fi
python -m pytest \$f -q -p no:cacheprovider > \$out/tests-file.log 2>&1; tail -n 1 \$out/tests-file.log
if [ -n "${QUICK:-}" ]; then echo GPU_DONE; exit 0; fi
python -m pytest tests/v1/sample --deselect \$f -q -p no:cacheprovider > \$out/tests-rest.log 2>&1; tail -n 1 \$out/tests-rest.log
if [ -n "${MEMCHECK:-}" ]; then
  compute-sanitizer --tool memcheck python -m pytest \$f -q -p no:cacheprovider -k "TestCudaTopkToppExact and not graph" > \$out/memcheck.log 2>&1
  tail -n 1 \$out/memcheck.log; grep -c "Invalid\|out of bounds" \$out/memcheck.log
fi
for pass in 0 1 2; do
  python benchmarks/benchmark_topk_topp.py --batch-sizes 1 4 8 16 64 128 256 512 1024 2048 \
    --vocab-sizes 131072 248320 --logits-file \$HOME/code/qwen36-verify-logits-t1.pt \
    --implementations triton cuda --check --output \$out/pass-\$pass.jsonl > \$out/pass-\$pass.log 2>&1
  echo "pass \$pass done"
done
for pass in 0 1 2; do
  python \$HOME/code/topk-research/large_k.py \$out/large-k-\$pass.jsonl > \$out/large-k-\$pass.log 2>&1
done
echo "large k done"
if [ -n "${RELATED:-}" ]; then bash \$HOME/code/topk-research/related_prs.sh 8af2080e71 \$out/related; fi
echo GPU_DONE
EOF
}

main
