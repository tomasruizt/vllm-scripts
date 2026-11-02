#!/usr/bin/env bash
# Runs one validation task for PR #65175 under `chg run`. The PR is Python-only,
# so a detached checkout in ~/code/vllm-xpress-av shares the compiled extensions
# of the ~/code/vllm-dsv41 build through untracked .so symlinks.
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT

export HF_HUB_CACHE=${HF_CACHE:-/mnt/nfs-preprod-1/engine/hub_cache}
export XDG_CACHE_HOME=$HOME/cache
export TMPDIR=$HOME/tmp
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$HOME/code/vllm-dsv41/.venv/bin:$CUDA_HOME/bin:$PATH"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
REPO_NAME=${REPO_NAME:-vllm-xpress-av}
export REPO=$HOME/code/$REPO_NAME
export PY=$HOME/code/vllm-dsv41/.venv/bin/python
export PYTHONPATH=$REPO
export PR_COMMIT=${PR_COMMIT:-e2d76c45bf}
export BASE_COMMIT=${BASE_COMMIT:-f9db1756e9}
export PORT=${PORT:-8640}
export EXPERIMENT=${EXPERIMENT:-$REPO_NAME}

echo "repo=$REPO commit=$(git -C "$REPO" rev-parse --short HEAD) port=$PORT" > env.txt
git -C "$REPO" status --short --untracked-files=no >> env.txt

# perf sweeps pin their own checkout (PR worktree vs base worktree), the rest
# start from the PR head.
checkout=$([[ $TASK == steady ]] && echo "" || echo "git -C $REPO checkout -q --detach $PR_COMMIT")
chg run ${CHG_GPUS:---gpu-ids 0} --wait 3h --note "xpress av 65175 $TASK" -- bash -c "
    set -euo pipefail
    cd $ROOT
    $checkout
    bash $ROOT/run-$TASK.sh > $TASK.log 2>&1
" || echo "$?" > chg-exit-code.txt
