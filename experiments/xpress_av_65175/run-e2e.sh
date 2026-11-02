#!/usr/bin/env bash
# Both variants (AV off, AV on) at main and at the PR head, sequentially on one
# GPU. BASE_COMMIT lacks return_logits, so the AV-on run there is the "estimator
# created but never fed" behaviour the PR fixes.
set -euo pipefail
for commit in "$BASE_COMMIT" "$PR_COMMIT"; do
    git -C "$REPO" checkout -q --detach "$commit"
    echo "=== $commit $(git -C $REPO log --oneline -1)" >> variants.log
    VLLM_COMMIT=$commit "$PY" "$ROOT/av_e2e.py" > "$commit.log" 2>&1 || echo "$commit exit $?" >> failures.txt
done
git -C "$REPO" checkout -q --detach "$PR_COMMIT"
grep -h '^RESULT ' ./*.log | sed 's/^RESULT //' > results.jsonl
grep -h '^commit=' ./*.log >> variants.log
