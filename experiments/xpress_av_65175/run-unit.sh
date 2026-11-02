#!/usr/bin/env bash
# Unit tests for the PR head. Fails-before does not apply: return_logits does not
# exist on main, so the test cannot even be collected there.
set -euo pipefail
cd "$REPO"
"$PY" -m pytest -v tests/v1/spec_decode/test_xpress_head.py
