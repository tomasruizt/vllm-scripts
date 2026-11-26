#!/usr/bin/env bash
# vLLM server for AgentX on GLM-5.3-Flash: TP4, native FP8 weights, FP8 KV cache, MTP 5, prefix caching on (default).
# Mirrors the SGLang InferenceX recipe (TP4, no EP, 1M context, MTP 5 steps) and the vLLM recipe's TP4 command.
# AgentX pins acceptance: 3.60 is InferenceX's golden AL for GLM-5.3-Flash MTP with 5 speculative tokens.
set -euo pipefail

SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":5,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.60}'

# Host-memory KV offloading, only when OFFLOAD_GIB_PER_RANK is set (the SGLang InferenceX recipe turns HiCache on from c40).
# Connector and eager mode as in InferenceX's vLLM agentic recipe for Kimi K3.1, the other KDA model.
offload=()
if [[ -n ${OFFLOAD_GIB_PER_RANK:-} ]]; then
    offload_bytes=$((OFFLOAD_GIB_PER_RANK << 30))
    offload_extra="{\"cpu_bytes_to_use_per_rank\":$offload_bytes,\"lazy_offload\":false}"
    offload=(--kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":$offload_extra}")
fi

export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_OFFLINE=1

exec vllm serve zai-org/GLM-5.3-Flash \
    --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c \
    --tokenizer-mode glm5_next \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --language-model-only \
    --speculative-config "$SPEC_CONFIG" \
    --tool-call-parser glm47 \
    --reasoning-parser glm47 \
    --enable-auto-tool-choice \
    --disable-uvicorn-access-log \
    --port 8000 \
    "${offload[@]}" \
    ${EXTRA_SERVER_ARGS:-}
