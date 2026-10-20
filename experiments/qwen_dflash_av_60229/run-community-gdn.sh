#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.venv/bin:$HOME/.local/bin:$PATH"
export MODEL=/data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
root="$HOME/runs/qwen-gdn-community-20261019"
cd "$HOME/runs/qwen-dflash-av-60229"
for arm in community gdn; do
    checkout="$HOME/code/vllm-av-community-base"
    if [[ "$arm" == gdn ]]; then
        checkout="$HOME/code/vllm-av-gdn-community"
    fi
    export PYTHONPATH="$checkout"
    export RAW_CURVE_OUTPUT="$root/$arm.json"
    "$HOME/.venv/bin/python" profile_community_gdn.py > "$root/$arm.log" 2>&1
done
touch "$root/DONE"
