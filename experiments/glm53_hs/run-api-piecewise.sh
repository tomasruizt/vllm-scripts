#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HS_GRAPH_MODE=PIECEWISE
exec bash "$SCRIPT_DIR/run-api.sh" "$@"
