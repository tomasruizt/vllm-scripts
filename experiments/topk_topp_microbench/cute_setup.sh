#!/usr/bin/env bash
# Own worktree for the CuTe DSL top-k/top-p filter (rh-xhuu), next to the CUDA PR trees other
# agents benchmark in. Main at BASE plus the files of PATCH (a tarball of the changed files),
# precompiled artifacts symlinked from main, and a copy (snapshot) of the CUDA tree's built
# _C_stable_libtorch with the CUDA filter op. Usage, on the GPU host: cute_setup.sh PATCH CUDA_TREE
set -euo pipefail
PATCH=$1 CUDA_TREE=$2
ROOT=$HOME/code
MAIN=$ROOT/topk-cuda-main
TREE=${TREE:-$ROOT/topk-cute}
BASE=8af2080e71

main() {
    [[ -d $TREE ]] || git -C "$MAIN" worktree add -q --detach "$TREE" "$BASE"
    git -C "$TREE" checkout -q --detach "$BASE"
    tar -xzf "$PATCH" -C "$TREE"
    link_artifacts
    cp "$CUDA_TREE/vllm/_C_stable_libtorch.abi3.so" "$TREE/vllm/_C_stable_libtorch.abi3.so.tmp"
    mv -f "$TREE/vllm/_C_stable_libtorch.abi3.so.tmp" "$TREE/vllm/_C_stable_libtorch.abi3.so"
    echo SETUP_OK
}

link_artifacts() {
    local path
    while IFS= read -r path; do
        [[ -e $TREE/$path || -L $TREE/$path ]] || { mkdir -p "$(dirname "$TREE/$path")"; ln -s "$MAIN/$path" "$TREE/$path"; }
    done < <(git -C "$MAIN" ls-files --others --exclude-standard --directory --ignored vllm | sed 's|/$||' | grep -v __pycache__ | grep -v _C_stable_libtorch)
}

main "$@"
