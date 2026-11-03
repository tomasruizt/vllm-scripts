#!/usr/bin/env bash
set -euo pipefail
root=${ROOT:?set ROOT to a fresh run directory}
runtime=${RUNTIME:?set RUNTIME to a precompiled vLLM checkout}
cache=${HF_HUB_CACHE:?set HF_HUB_CACHE to a writable shared cache}
here=$(cd "$(dirname "$0")" && pwd)
head=f94d7ff8ae9ab663e4e8b08f9dd0f0326bad0b16
base=1cb31be7028457c7eae7b6fabca1941d5d3c4730
resolver=509acae921bad5bcb6a1fb96a8cc5ceb4fb3f6b2
speculator=vllm/v1/worker/gpu/spec_decode/dspark/speculator.py
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$runtime/.venv/bin:$CUDA_HOME/bin:$PATH

main() {
    [[ ! -e $root/source ]] || { echo "Use a fresh ROOT: $root/source exists" >&2; exit 1; }
    [[ -d $cache && -w $cache ]] || { echo "Shared cache is not writable: $cache" >&2; exit 1; }
    mkdir -p "$root/source/pr" "$root/source/base" "$root/speed_loader"
    root=$(cd "$root" && pwd)
    trap 'echo $? > "$root/setup-exit.txt"' EXIT
    if [[ $here != "$root" ]]; then cp "$here"/{run.sh,sweep.yaml,check_outputs.py,prepare_data.py} "$root/"; fi
    curl -fL --retry 3 "https://api.github.com/repos/mugglewei97/vllm/tarball/$head" -o "$root/pr-$head.tar.gz"
    tar -xzf "$root/pr-$head.tar.gz" --strip-components=1 -C "$root/source/pr"
    cp -a "$root/source/pr/." "$root/source/base/"
    curl -fL --retry 3 "https://raw.githubusercontent.com/vllm-project/vllm/$base/$speculator" -o "$root/source/base/$speculator"
    for variant in pr base; do attach_runtime "$variant"; done
    uv venv --python 3.12 "$root/client-venv"
    uv pip install --python "$root/client-venv/bin/python" aiperf==0.13.0
    prepare_data
    cd "$root"
    PYTHONPATH="$root/source/pr" "$runtime/.venv/bin/python" -c 'from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator; import vllm; print(vllm.__file__)'
}

attach_runtime() {
    local variant=$1
    local file rel
    while IFS= read -r file; do
        rel=${file#"$runtime/"}
        mkdir -p "$root/source/$variant/$(dirname "$rel")"
        ln -s "$file" "$root/source/$variant/$rel"
    done < <(find "$runtime/vllm" -name '*.so')
    cp -an "$runtime/vllm/third_party/." "$root/source/$variant/vllm/third_party/"
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

main "$@"
