#!/usr/bin/env bash
# The new AV case alone on main (must fail on acceptance length), then the same
# test plus the existing case on the PR head (must pass). test-overrides.patch
# adds the case and the ArgsTest field, so main runs the new test against main's
# source while the PR runs it against the PR's source.
set -euo pipefail
cd "$REPO"
git apply "$ROOT/test-overrides.patch"

run_tests() {  # selector, log path
    "$PY" -m pytest -v -s tests/v1/e2e/spec_decode/acceptance_rates/xpress/ \
        -k "$1" > "$2" 2>&1 || echo "$2 failed" >> "$ROOT/failures.txt"
}

git checkout -q --detach "$BASE_COMMIT"
run_tests qwen3_xpress_av "$ROOT/$BASE_COMMIT-test.log"
git checkout -q --detach "$PR_COMMIT"
run_tests qwen3_xpress "$ROOT/$PR_COMMIT-test.log"
