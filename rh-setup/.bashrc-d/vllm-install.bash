# Install the current checkout with the latest published CUDA 13.0 wheel.
unalias vllm-install 2>/dev/null || true
vllm-install() {
    local wheel_url
    wheel_url=$(python - <<'PY'
import json
import platform
from urllib.parse import urljoin
from urllib.request import urlopen

base = "https://wheels.vllm.ai/nightly/cu130/vllm/"
with urlopen(base + "metadata.json", timeout=30) as response:
    wheels = json.load(response)
matches = [w for w in wheels if w["platform_tag"].endswith("_" + platform.machine())]
if len(matches) != 1:
    raise SystemExit("Expected one nightly CUDA 13.0 wheel for " + platform.machine())
print(urljoin(base, matches[0]["path"]))
PY
    ) || return
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$wheel_url" \
        uv pip install --editable . --torch-backend=auto "$@"
}
