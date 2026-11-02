#!/usr/bin/env bash
# Run all variants sequentially on one GPU under `chg run` (released when it exits).
set -euo pipefail
export ROOT=${ROOT:?}
cd "$ROOT"
trap 'printf "%s\n" "$?" > launch-exit-code.txt' EXIT
export HF_HUB_CACHE=/mnt/nfs-preprod-1/engine/hub_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$HOME/cache
export TMPDIR=$HOME/tmp
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1  # collective_rpc with a function
VLLM_PYTHON=${VLLM_PYTHON:-$HOME/code/vllm-pr-63205/.venv/bin/python}
VARIANTS=${VARIANTS:-auto,trained,online,no_av,online_argmax,trained_argmax,xpress,xpress_av}
VLLM_REPO=$(dirname "$(dirname "$(dirname "$VLLM_PYTHON")")")
git -C "$VLLM_REPO" log --oneline -1 > vllm-commit.txt
git -C "$VLLM_REPO" diff > vllm-local.patch
chg run ${CHG_GPUS:---gpus 1} --wait 2h --note "DSpark online estimator smoke test" -- bash -c "
    (cd $VLLM_REPO && $VLLM_PYTHON -m pytest -q --noconftest tests/v1/spec_decode/test_xpress_head.py tests/test_config.py::test_dspark_adaptive_verification_separates_graph_cache tests/test_config.py::test_dspark_adaptive_verification_uses_draft_logits) > pytest.log 2>&1 || echo \"pytest exit \$?\" >> failures.txt
    for v in ${VARIANTS//,/ }; do $VLLM_PYTHON smoke.py \$v > \$v.log 2>&1 || echo \"\$v exit \$?\" >> failures.txt; done"
grep -h '^RESULT ' ./*.log | sed 's/^RESULT //' > results.jsonl
