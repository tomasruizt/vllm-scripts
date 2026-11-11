#!/usr/bin/env bash
# Build the top-k/top-p filter PR on a GPU host: upstream main with precompiled
# wheels, plus a worktree with the PR files (a tarball, or a .patch applied with git) where
# _C_stable_libtorch is built from source. Usage: pr_build.sh BASE_COMMIT ARCH PATCH
set -euo pipefail
BASE=$1 ARCH=$2 PATCH=$3
ROOT=$HOME/code
MAIN=${MAIN:-$ROOT/topk-cuda-main}
TREE=${TREE:-$ROOT/topk-cuda-pr}
GCC=${GCC:-/usr/bin}
export TORCH_CUDA_ARCH_LIST=$ARCH PATH=/usr/local/cuda-13.0/bin:$PATH

main() {
    setup_main
    setup_tree
    build
    echo BUILD_OK
}

setup_main() {
    [[ -d $MAIN ]] || git clone -q --filter=blob:none https://github.com/vllm-project/vllm.git "$MAIN"
    cd "$MAIN"
    git fetch -q origin main
    git checkout -q --detach "$BASE"
    [[ -x .venv/bin/python ]] || uv venv -q --python 3.12 .venv
    VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 uv pip install -q --python .venv/bin/python -e . --torch-backend=auto
    uv pip install -q --python .venv/bin/python -r requirements/test/cuda.in cmake ninja
}

# Precompiled artifacts are untracked in main, so the worktree gets symlinks to them.
setup_tree() {
    [[ -d $TREE ]] || git -C "$MAIN" worktree add -q --detach "$TREE" "$BASE"
    git -C "$TREE" checkout -q --detach "$BASE"
    git -C "$TREE" checkout -q -- .
    case $PATCH in
        *.patch) git -C "$TREE" apply --3way "$PATCH" ;;
        *) tar -xzf "$PATCH" -C "$TREE" ;;
    esac
    local path
    while IFS= read -r path; do
        [[ -e $TREE/$path ]] || { mkdir -p "$(dirname "$TREE/$path")"; ln -s "$MAIN/$path" "$TREE/$path"; }
    done < <(untracked --ignored; untracked)
}

untracked() {
    git -C "$MAIN" ls-files --others --exclude-standard --directory "$@" vllm | sed 's|/$||' | grep -v __pycache__
}

build() {
    export PATH=$MAIN/.venv/bin:$PATH
    cd "$TREE"
    rm -f vllm/_C_stable_libtorch*.so
    cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
        -DVLLM_PYTHON_EXECUTABLE="$MAIN/.venv/bin/python" -DCMAKE_INSTALL_PREFIX="$PWD" \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DNVCC_THREADS=4 \
        -DCMAKE_C_COMPILER=$GCC/gcc -DCMAKE_CXX_COMPILER=$GCC/g++ -DCMAKE_CUDA_HOST_COMPILER=$GCC/g++ > cmake.log
    cmake --build cmake-build-release --target _C_stable_libtorch -j 64 >> cmake.log
    cmake --install cmake-build-release --component _C_stable_libtorch >> cmake.log
}

main "$@"
