#!/usr/bin/env bash
# One GLM-5.3-Flash E2E variant on MI355X, running run.sh inside the ROCm vLLM image.
# Usage, on the AMD host from the run dir: chg run --gpus 4 -- bash amd_e2e.sh VARIANT
# main-*: the image as is. hier-*: the PR's vllm files (overlay/, rebased on the image's
# commit) mounted over the image's. Run amd_setup.sh once per run dir first.
set -euo pipefail
image=${IMAGE:-docker.io/vllm/vllm-openai-rocm:nightly-rocm100-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf}
site=/usr/local/lib/python3.12/dist-packages
mounts=(-v "$PWD:/work:z" -v "$HOME/hf_hub:/hub:z")
if [[ $1 == hier-* ]]; then
    overlay=${OVERLAY:-overlay}  # e.g. a patched copy of overlay/
    while read -r f; do mounts+=(-v "$PWD/$overlay/$f:$site/$f:ro,z"); done < "$overlay/files.txt"
fi
# HIP numbers the GPUs in another order than amd-smi and chg: by render node (renderD128
# is HIP 0, then every 8th minor). Map chg's GPUs to HIP's by PCI address.
hip=()
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
    bdf=$(amd-smi list | awk -v g="GPU: $gpu" '$0 == g {getline; print $2}')
    render=$(readlink -f "/dev/dri/by-path/pci-$bdf-render")
    hip+=($(( (${render#/dev/dri/renderD} - 128) / 8 )))
done
hip_devices=$(IFS=,; echo "${hip[*]}")
# Env and attention backend of the GLM-5.3-Flash MI355X recipe (recipes.vllm.ai), which
# validates MTP5 at TP=4. It sets no FP8 KV cache.
recipe=(-e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
    -e VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1
    -e "GLM_SERVER_ARGS=--attention-backend ROCM_AITER_MLA_SPARSE${EXTRA_SERVER_ARGS:+ $EXTRA_SERVER_ARGS}")
env_args=("${recipe[@]}" -e HIP_VISIBLE_DEVICES="$hip_devices" -e HOME=/work/home -e TRITON_CACHE_DIR=/work/home/triton
    -e HUB_CACHE=/hub -e IMAGE="$image" -e PRESET=glm -e VENV_OVERRIDE=/work/py -e SRC_OVERRIDE= -e MAIN=/work/src
    -e AIPERF=/work/aiperf-venv/bin/aiperf -e DATASET=/work/throughput_1k.jsonl -e AIME_DATA=/work/aime25.jsonl)
MAX_MODEL_LEN=131072  # AIME needs long generations
for name in MAX_MODEL_LEN AIME_N REPEATS GSM8K_QUESTIONS CONCURRENCIES DURATION; do
    [[ -z ${!name+x} ]] || env_args+=(-e "$name=${!name}")
done
exec podman run --rm ${PODMAN_NAME:+--name "$PODMAN_NAME"} --device /dev/kfd --device /dev/dri --ipc=host --security-opt seccomp=unconfined \
    "${env_args[@]}" "${mounts[@]}" -w /work --entrypoint bash "$image" ${RUN:-run.sh} "$@"
