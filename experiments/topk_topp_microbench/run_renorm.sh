#!/usr/bin/env bash
# Runs the masked-logit checks, then the timing sweep, on one reserved GPU.
set -euo pipefail
python=$1
shift
"$python" validate_renorm.py --output results/validation.jsonl
"$python" sweep_renorm.py --output results/timing "$@"
