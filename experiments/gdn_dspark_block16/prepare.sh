#!/usr/bin/env bash
set -euo pipefail
root="${RUN_ROOT:-$HOME/runs/gdn-dspark-block16-20261026-1745}"
export PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH"
export HF_HUB_CACHE=/data/tomasruizt/hub_cache
mkdir -p "$root"
trap 'printf "%s\n" "$?" > "$root/prepare-exit.txt"' EXIT
hf download "${TARGET_MODEL_ID:-Qwen/Qwen3.6-27B}" --revision "${TARGET_REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}" \
    --cache-dir "$HF_HUB_CACHE" > "$root/target-download.log" 2>&1 &
target_download=$!
hf download "${DRAFT_MODEL_ID:-satgeze/Qwen3.6-27B-DSpark}" --revision "${DRAFT_REVISION:-47d8de04dd814f4a727bf3fc6a85613d2a3dcaa1}" \
    --cache-dir "$HF_HUB_CACHE" --exclude '*.gguf' > "$root/draft-download.log" 2>&1 &
draft_download=$!
wait "$target_download"
wait "$draft_download"
touch "$root/MODELS_READY"
