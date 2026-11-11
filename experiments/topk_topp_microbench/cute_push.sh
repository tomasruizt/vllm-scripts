#!/usr/bin/env bash
# Sync the local CuTe DSL worktree (files changed since BASE) to the GPU host's own tree.
# Usage: cute_push.sh [HOST] [CUDA_TREE]
set -euo pipefail
host=${1:-RH-nm-ore-b300-01} cuda_tree=${2:-\$HOME/code/topk-cuda-pr2}
local_tree=${LOCAL_TREE:-$HOME/code/rh/vllm-topk-cute}
here=$(cd "$(dirname "$0")" && pwd)
ssh_=(ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none "$host")
tarball=$(mktemp -t cute-patch).tgz
(cd "$local_tree" && { git diff --name-only 8af2080e71; git ls-files --others --exclude-standard vllm tests benchmarks; } | sort -u | grep -v __pycache__ | COPYFILE_DISABLE=1 tar --no-xattrs -czf "$tarball" -T -)
scp -q -o BatchMode=yes "$tarball" "$host:cute-patch.tgz"
scp -q -o BatchMode=yes "$here/cute_setup.sh" "$host:cute_setup.sh"
"${ssh_[@]}" "TREE=${TREE:-\$HOME/code/topk-cute} bash cute_setup.sh \$HOME/cute-patch.tgz $cuda_tree"
rm -f "$tarball"
