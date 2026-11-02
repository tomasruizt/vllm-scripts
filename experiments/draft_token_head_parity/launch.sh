#!/usr/bin/env bash
# Unit tests at HEAD, then every variant at BASE_COMMIT and HEAD_COMMIT, sequentially on one GPU under
# `chg run`. Both commits share the editable install: their diff is Python-only.
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export HF_HUB_CACHE=${HF_CACHE:-/mnt/nfs-preprod-1/engine/hub_cache}
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$HOME/cache
export TMPDIR=$HOME/tmp
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1  # collective_rpc with a function
VLLM_PYTHON=${VLLM_PYTHON:-$HOME/code/vllm-pr-63205/.venv/bin/python}
VLLM_REPO=$(dirname "$(dirname "$(dirname "$VLLM_PYTHON")")")
BASE_COMMIT=${BASE_COMMIT:-d33fcbac27}  # Online acceptance estimator for DSpark and XPress (before the refactor)
HEAD_COMMIT=${HEAD_COMMIT:-d5c00eb0ff}  # Draft token heads for block-diffusion speculators
VARIANTS=${VARIANTS:-dflash_greedy,dflash_greedy_av,dflash_prob,xpress_greedy,xpress_greedy_av,dspark_av,dspark_online}
TESTS=${TESTS:-tests/v1/spec_decode/test_custom_speculator.py tests/v1/spec_decode/test_xpress_head.py tests/v1/spec_decode/test_dflash2.py tests/v1/spec_decode/test_acceptance_estimator.py tests/watermarking/test_watermarking.py tests/v1/worker/test_dspark_utils.py tests/test_config.py}
COMMITS=${COMMITS:-$BASE_COMMIT $HEAD_COMMIT}  # each runs every variant
git -C "$VLLM_REPO" status --short > vllm-status.txt
chg run ${CHG_GPUS:---gpus 1} --wait 2h --note "draft token head parity" -- bash -c "
    git -C $VLLM_REPO checkout -q --detach $HEAD_COMMIT
    (cd $VLLM_REPO && $VLLM_PYTHON -m pytest -q $TESTS) > pytest.log 2>&1 || echo \"pytest exit \$?\" >> failures.txt
    for c in $COMMITS; do
        git -C $VLLM_REPO checkout -q --detach \$c
        for v in ${VARIANTS//,/ }; do
            VLLM_COMMIT=\$c $VLLM_PYTHON parity.py \$v > \$c-\$v.log 2>&1 || echo \"\$c \$v exit \$?\" >> failures.txt
        done
    done
    git -C $VLLM_REPO checkout -q --detach $HEAD_COMMIT"
grep -h '^RESULT ' ./*.log | sed 's/^RESULT //' > results.jsonl
