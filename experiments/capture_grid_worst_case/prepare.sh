#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data/tomasruizt/runs/capture-worst-case-20261020}
cd "$ROOT"
trap 'printf "%s\n" "$?" > prepare-exit-code.txt' EXIT
export HF_HUB_CACHE=/data/tomasruizt/hub_cache
export HF_XET_CACHE=/data/tomasruizt/xet-cache
export TMPDIR=/data/tomasruizt/tmp
mkdir -p "$TMPDIR"
for item in Qwen3-32B:9216db5781bf21249d130ec9da846c4624c16137 Qwen3.5-27B:fc05daec18b0a78c049392ed2e771dde82bdf654; do
    model=${item%%:*}
    revision=${item#*:}
    "$HOME/.venv/bin/hf" download "Qwen/$model" --revision "$revision" --cache-dir "$HF_HUB_CACHE" > "$model-download.log" 2>&1
    "$HOME/.venv/bin/python" - "$HF_HUB_CACHE/models--Qwen--$model/snapshots/$revision" <<'PY'
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
index = json.loads((p / "model.safetensors.index.json").read_text())
files = set(index["weight_map"].values())
assert all((p / name).is_file() for name in files)
print(f"Verified {len(files)} shards for {p.name}")
PY
done
