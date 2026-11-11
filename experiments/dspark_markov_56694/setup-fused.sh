#!/usr/bin/env bash
# Build the fused-revision validation trees for PR #56694 (head 0e5254884c).
# Installs the exact per-commit wheel for the PR merge base, then overlays the
# three PR Python files. The unfused control is the same tree with
# can_fuse_markov_walk forced to False.
set -euo pipefail
root=${ROOT:?set ROOT to a fresh run directory}
cache=${HF_HUB_CACHE:?set HF_HUB_CACHE to a writable cache}
head=0e5254884ceb71fb43919533c4708c6eb934117a
base=1cb31be7028457c7eae7b6fabca1941d5d3c4730
wheel=https://wheels.vllm.ai/$base/cu130
resolver=509acae921bad5bcb6a1fb96a8cc5ceb4fb3f6b2
here=$(cd "$(dirname "$0")" && pwd)

main() {
    [[ -d $cache && -w $cache ]] || { echo "Cache is not writable: $cache" >&2; exit 1; }
    root=$(cd "$root" && pwd)
    [[ -n $root && $root != / ]] || { echo "Refusing to clean $root" >&2; exit 1; }
    rm -rf "$root/source" "$root/prsrc"
    mkdir -p "$root/source" "$root/speed_loader"
    if [[ $here != "$root" ]]; then
        cp "$here"/{run-fused.sh,sweep-fixed.yaml,check_outputs.py,prepare_data.py} "$root/"
        cp "$here/../../benchmarking/host_load.sh" "$root/"
    fi
    install_venv
    build_tree pr
    overlay_pr
    build_derived_trees
    install_client
    prepare_data
    smoke
    touch "$root/setup-complete.txt"
}

install_venv() {
    uv venv --python 3.12 "$root/venv"
    uv pip install --python "$root/venv/bin/python" --pre "vllm==0.33.0rc1.dev168+g1cb31be70" \
        --extra-index-url "$wheel" --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match
    uv pip install --python "$root/venv/bin/python" pytest
}

# Copy the wheel's compiled package, keeping .so files in place.
build_tree() {
    mkdir -p "$root/source/$1"
    cp -a "$root/venv/lib/python3.12/site-packages/vllm" "$root/source/$1/vllm"
}

overlay_pr() {
    curl -fL --retry 3 "https://api.github.com/repos/mugglewei97/vllm/tarball/$head" -o "$root/pr.tar.gz"
    mkdir -p "$root/prsrc"
    tar -xzf "$root/pr.tar.gz" --strip-components=1 -C "$root/prsrc"
    cp -a "$root/prsrc/vllm/." "$root/source/pr/vllm/"
}

build_derived_trees() {
    local dspark=vllm/v1/worker/gpu/spec_decode/dspark
    for variant in pr-unfused pr-instrumented; do
        mkdir -p "$root/source/$variant"
        cp -a "$root/source/pr/vllm" "$root/source/$variant/vllm"
    done
    disable_fusion "$root/source/pr-unfused/$dspark/topk_markov.py"
    add_fusion_log "$root/source/pr-instrumented/$dspark/speculator.py"
}

disable_fusion() {
    python3 - "$1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = '    """Whether the greedy kernel supports this Markov head and logits."""\n'
assert old in s
p.write_text(s.replace(old, old + "    return False  # local unfused control\n", 1))
PY
}

add_fusion_log() {
    python3 - "$1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = "                markov_walk_topk(\n"
assert old in s
p.write_text(s.replace(old, '                logger.warning_once("FUSED_MARKOV_WALK_ACTIVE")\n' + old, 1))
PY
}

install_client() {
    uv venv --python 3.12 "$root/client-venv"
    uv pip install --python "$root/client-venv/bin/python" aiperf==0.13.0
}

prepare_data() {
    "$root/client-venv/bin/hf" download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --cache-dir "$cache"
    "$root/client-venv/bin/hf" download AQ-MedAI/Qwen3-4B-DSpark --revision 49d6beedccdb7e2500382171461cb9868c2c5fbc --cache-dir "$cache"
    for bucket in 8k 16k; do
        "$root/client-venv/bin/hf" download nvidia/SPEED-Bench "throughput_$bucket/test-00000-of-00001.parquet" --repo-type dataset --revision fbd6dc64353a3153bf24252a113ebd6d2cff9dec --local-dir "$root/data/speed"
    done
    for name in base speed; do
        curl -fL --retry 3 "https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/$resolver/examples/specdec_bench/specdec_bench/datasets/$name.py" -o "$root/speed_loader/$name.py"
    done
    touch "$root/speed_loader/__init__.py"
    "$root/client-venv/bin/python" "$root/prepare_data.py" "$root"
}

smoke() {
    PYTHONPATH="$root/source/pr" "$root/venv/bin/python" - <<'PY'
import torch, vllm
from vllm.v1.worker.gpu.spec_decode.dspark.topk_markov import can_fuse_markov_walk
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
print("vllm", vllm.__version__, vllm.__file__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("DSparkSpeculator", DSparkSpeculator.__module__)
PY
}

main "$@"
