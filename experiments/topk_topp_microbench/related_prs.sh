#!/usr/bin/env bash
# Benchmark the Triton top-k/top-p PRs #62596 and #65004 against main's Triton kernel and the
# CUDA filter, on the same GPU in one session. Each PR's vllm/ diff is applied to main at BASE.
# Usage (on the GPU host, inside a chg reservation): related_prs.sh BASE OUT_DIR
set -euo pipefail
BASE=$1 OUT=$2
ROOT=$HOME/code
MAIN=$ROOT/topk-cuda-main
CUDA_TREE=$ROOT/topk-cuda-pr2
export PATH=$MAIN/.venv/bin:$PATH

main() {
    local pr
    for pr in 62596 65004; do setup_tree "$pr"; done
    mkdir -p "$OUT"
    for pass in 0 1 2; do
        bench main "$MAIN" triton "$pass"
        bench pr62596 "$ROOT/topk-pr62596" triton "$pass"
        bench pr65004 "$ROOT/topk-pr65004" triton "$pass"
        bench cuda "$CUDA_TREE" cuda "$pass"
        fastpath "$pass"
    done
    echo BENCH_DONE
}

# Main at BASE plus the PR's patch. Precompiled artifacts are untracked in main, so they are symlinked.
setup_tree() {
    local pr=$1 tree=$ROOT/topk-pr$1
    [[ -d $tree ]] || git -C "$MAIN" worktree add -q --detach "$tree" "$BASE"
    git -C "$tree" checkout -q --detach "$BASE"
    git -C "$tree" checkout -q -- .
    git -C "$tree" clean -fdq vllm  # files a previous patch added
    git -C "$tree" apply "$ROOT/pr$pr.patch"
    local path
    while IFS= read -r path; do
        [[ -e $tree/$path ]] || { mkdir -p "$(dirname "$tree/$path")"; ln -s "$MAIN/$path" "$tree/$path"; }
    done < <(git -C "$MAIN" ls-files --others --exclude-standard --directory --ignored vllm | sed 's|/$||' | grep -v __pycache__)
}

# FlashInfer fast path (#65649), in the CUDA tree, only on the scenarios it supports.
fastpath() {
    (cd "$CUDA_TREE" && PYTHONPATH=$CUDA_TREE:$ROOT/topk-research python "$ROOT/topk-research/fastpath_bench.py" \
        "$OUT/fastpath-pass-$1.jsonl" --logits-file "$ROOT/qwen36-verify-logits-t1.pt" > "$OUT/fastpath-pass-$1.log" 2>&1)
}

# The benchmark script and the CUDA wrapper come from the CUDA tree, so every arm runs the same
# workload. Without the custom op, the wrapper reports the kernel as unavailable.
bench() {
    local name=$1 tree=$2 impl=$3 pass=$4
    [[ -e $tree/vllm/v1/sample/ops/topk_topp_cuda.py ]] || cp "$CUDA_TREE/vllm/v1/sample/ops/topk_topp_cuda.py" "$tree/vllm/v1/sample/ops/"
    (cd "$tree" && PYTHONPATH=$tree python "$CUDA_TREE/benchmarks/benchmark_topk_topp.py" \
        --batch-sizes 1 4 8 16 64 128 256 512 1024 2048 --vocab-sizes 131072 248320 --logits-file "$ROOT/qwen36-verify-logits-t1.pt" \
        --implementations "$impl" --check --output "$OUT/$name-pass-$pass.jsonl" > "$OUT/$name-pass-$pass.log" 2>&1)
}

main "$@"
