# Use the GPUs currently reserved by this user.
set-cuda-visible() {
    local status devices
    status=$(canhazgpu status --json) || return
    devices=$(jq -r --arg user "$(id -un)" \
        '[.[] | select(.user == $user and .status == "IN_USE") | .gpu_id | tostring] | join(",")' \
        <<< "$status") || return
    export CUDA_VISIBLE_DEVICES="$devices"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
}
