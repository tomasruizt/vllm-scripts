#!/usr/bin/env bash
set -euo pipefail
ENV_DIR=/data/tomasruizt/capture-env
export UV_CACHE_DIR=/data/tomasruizt/cache/uv
if [[ ! -d $ENV_DIR ]]; then uv venv --python "$HOME/.venv/bin/python" "$ENV_DIR"; fi
printf "import site; site.addsitedir('%s/.venv/lib/python3.12/site-packages')\n" "$HOME" > "$ENV_DIR/lib/python3.12/site-packages/base-env.pth"
uv pip install --python "$ENV_DIR/bin/python" --no-deps --extra-index-url https://flashinfer.ai/whl flashinfer-python==0.7.0.post1 flashinfer-cubin==0.7.0.post1
