#!/usr/bin/env bash
# Kept-token prototype (compact top-k/top-p output) on a GPU host: the new and existing
# top-k/top-p and Gumbel tests, then kept_tokens_bench.py on real Qwen verify rows.
# Usage, on the host: kept_tokens.sh BASE_TREE COMMIT PATCH OUT
# BASE_TREE is a vllm checkout with precompiled artifacts and a .venv, which are reused.
set -euo pipefail
BASE=$1 COMMIT=$2 PATCH=$3 OUT=$4
TREE=$HOME/code/vllm-kept-tokens
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$BASE/.venv/bin:$PATH PYTHONPATH=$TREE

main() {
    setup_tree
    mkdir -p "$OUT"
    cd "$TREE"
    chg run --gpus 1 --timeout 40m --note kept-tokens -- bash -c "$(declare -f run); run '$OUT' '$HERE'"
    echo KEPT_DONE
}

run() {
    local out=$1 here=$2
    nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader > "$out/gpu.txt"
    python -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -p no:cacheprovider -k "kept" > "$out/tests-kept.log" 2>&1 || true
    python -m pytest tests/v1/worker/test_gpu_gumbel_sample.py -q -p no:cacheprovider > "$out/tests-gumbel.log" 2>&1 || true
    python -m pytest tests/v1/spec_decode/test_rejection_sampler_utils.py tests/v1/worker/test_gpu_rejection_sampler_chunking.py -q -p no:cacheprovider > "$out/tests-rejection.log" 2>&1 || true
    python -m pytest tests/v1/sample/test_topk_topp_sampler.py -q -p no:cacheprovider -k "TestTritonTopkTopp" > "$out/tests-triton.log" 2>&1 || true
    python -m pytest tests/v1/worker/test_gpu_sampler_flags.py tests/v1/worker/test_gpu_logits_processors.py tests/v1/worker/test_gpu_thinking_budget.py tests/v1/worker/test_gpu_rejection_sampler_i64.py -q -p no:cacheprovider > "$out/tests-worker.log" 2>&1 || true
    [[ -n ${SKIP_BENCH:-} ]] && return
    python "$here/kept_tokens_bench.py" "$out/bench.jsonl" --logits-file "$HOME/code/qwen36-verify-logits-t1.pt" > "$out/bench.log" 2>&1 || true
}

# Worktree at COMMIT plus the patch. Untracked precompiled artifacts are symlinked from BASE.
setup_tree() {
    git -C "$BASE" fetch -q https://github.com/vllm-project/vllm.git "$COMMIT"
    [[ -d $TREE ]] || git -C "$BASE" worktree add -q --detach "$TREE" "$COMMIT"
    git -C "$TREE" checkout -q -f --detach "$COMMIT"
    git -C "$TREE" apply "$PATCH"
    local path
    while IFS= read -r path; do
        [[ -e $TREE/$path ]] || { mkdir -p "$(dirname "$TREE/$path")"; ln -s "$BASE/$path" "$TREE/$path"; }
    done < <(git -C "$BASE" ls-files --others --exclude-standard --directory --ignored vllm | sed 's|/$||' | grep -v __pycache__)
}

main
