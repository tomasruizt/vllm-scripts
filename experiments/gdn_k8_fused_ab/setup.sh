#!/usr/bin/env bash
# Setup: main 5225a1f17f with precompiled wheels, plus worktrees with _C_stable_libtorch built from source.
# Defaults are for the H100. B300: MAIN=~/code/gdn-k8-verify-b300-main ARCH=10.3 GCC=/usr/bin FETCH=perf/gdn-fused-mtp-decode-k8-splitv WORKTREES=gdn-k8-splitv2:1e7e160b68
set -euo pipefail
ROOT=$HOME/code
MAIN=${MAIN:-$ROOT/gdn-k8-verify-h100-main}
FORK=https://github.com/tomasruizt/vllm.git
export TORCH_CUDA_ARCH_LIST=${ARCH:-9.0} PATH=/usr/local/cuda-13.0/bin:$PATH
GCC=${GCC:-/opt/rh/gcc-toolset-14/root/usr/bin}  # H100 system GCC 11 lacks C++20 <format>
WORKTREES=${WORKTREES:-gdn-k8-pr-on-main:2a9f894eab gdn-k8-splitv:52597d1936}
FETCH=${FETCH:-gdn-fused-mtp-decode-k8-main gdn-fused-mtp-decode-splitv}  # fork branches holding the worktree commits

main() {
    setup_main
    local worktree
    for worktree in $WORKTREES; do add_worktree "$ROOT/${worktree%%:*}" "${worktree#*:}"; done
    echo SETUP_OK
}

setup_main() {
    [[ -d $MAIN ]] || git clone -q --filter=blob:none https://github.com/vllm-project/vllm.git "$MAIN"
    cd "$MAIN"
    git fetch -q "$FORK" $FETCH
    git checkout -q --detach 5225a1f17f
    [[ -x .venv/bin/python ]] || uv venv -q --python 3.12 .venv
    VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=13.0 uv pip install -q --python .venv/bin/python -e . --torch-backend=auto
    uv pip install -q --python .venv/bin/python -r requirements/test/cuda.in cmake ninja
}

add_worktree() {
    local dir=$1 commit=$2
    [[ -d $dir ]] || git -C "$MAIN" worktree add -q --detach "$dir" "$commit"
    git -C "$dir" checkout -q --detach "$commit"
    link_artifacts "$dir"
    build "$dir"
}

# Precompiled artifacts are untracked in main, so the worktree gets symlinks to them.
link_artifacts() {
    local dir=$1 path
    while IFS= read -r path; do
        [[ -e $dir/$path ]] || { mkdir -p "$(dirname "$dir/$path")"; ln -s "$MAIN/$path" "$dir/$path"; }
    done < <(untracked "$MAIN" --ignored; untracked "$MAIN")
}

untracked() {
    git -C "$1" ls-files --others --exclude-standard --directory "${@:2}" vllm | sed 's|/$||' | grep -v __pycache__
}

build() {
    local dir=$1
    export PATH=$MAIN/.venv/bin:$PATH
    cd "$dir"
    rm -rf vllm/_C_stable_libtorch*.so cmake-build-release
    cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
        -DVLLM_PYTHON_EXECUTABLE="$MAIN/.venv/bin/python" -DCMAKE_INSTALL_PREFIX="$PWD" \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DNVCC_THREADS=4 \
        -DCMAKE_C_COMPILER=$GCC/gcc -DCMAKE_CXX_COMPILER=$GCC/g++ -DCMAKE_CUDA_HOST_COMPILER=$GCC/g++ > "$dir/cmake.log"
    cmake --build cmake-build-release --target _C_stable_libtorch -j 64 >> "$dir/cmake.log"
    cmake --install cmake-build-release --component _C_stable_libtorch >> "$dir/cmake.log"
}

main "$@"
