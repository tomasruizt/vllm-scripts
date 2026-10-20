#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.venv/bin:$HOME/.local/bin:$PATH"
export MODEL=/data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024
root="$HOME/runs/qwen-av-tail-profile-20261019"
cd "$HOME/runs/qwen-dflash-av-60229"
for arm in slot_only mixed; do
    checkout="$HOME/code/vllm-av-gdn-profile"
    if [[ "$arm" == mixed ]]; then
        checkout="$HOME/code/vllm-av-tail-profile"
    fi
    export PYTHONPATH="$checkout"
    export TAIL_ARM="$arm"
    export TAIL_OUTPUT="$root/$arm"
    export RAW_CURVE_OUTPUT="$root/$arm.json"
    "$HOME/.venv/bin/python" -c 'import vllm; print(vllm.__file__, flush=True)'
    "$HOME/.venv/bin/python" profile_tail_batches.py > "$root/$arm.log" 2>&1
done
echo TAIL_PROFILES_DONE
