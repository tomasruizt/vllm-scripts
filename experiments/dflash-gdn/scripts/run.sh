#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
engine_python=${ENGINE_PYTHON:-"$HOME/.venv/bin/python"}
exec canhazgpu run --gpus 1 -- "$engine_python" "$script_dir/run.py" "$@"
