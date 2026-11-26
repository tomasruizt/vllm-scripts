#!/usr/bin/env bash
# Step-time microbenchmark in the GPU pod: MTP 5 verify steps (6 rows per request) at growing batch sizes.
# `vllm bench forward` does not engage the specialized decode, so every case times the stock path. Still useful for the
# scaling with batch size, and for the cost of a prefill chunk that shares a step with decoding requests.
# Same model flags as serve.sh, real weights. Env: RUN_ID, and for smoke tests on the small clone MODEL, REVISION, TP and
# LOAD_FORMAT (the clone has no MTP weights, so dummy).
set -uo pipefail

OUT=/models/results/$RUN_ID
mkdir -p "$OUT"
cp /scripts/*.sh "$OUT/"
vllm --version > "$OUT/vllm-version.txt" 2>&1
export HF_HUB_OFFLINE=1
MODEL=${MODEL:-zai-org/GLM-5.3-Flash}
REVISION=${REVISION-46a050bec13002ad29ee795c16e61cf4fa3c4a9c}
TP=${TP:-4}
LOAD_FORMAT=${LOAD_FORMAT:-auto}

SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":5,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.60}'
# Short context and a low memory utilization: the default settings ran out of GPU memory from 11 requests on.
KV=30000
cases=()
for n in 1 4 8 10 11 12 16 24 32; do
    cases+=(--case "verify$n=$n*6@$KV:5")
done
# Prompt rows in the step: 8 verifying requests plus one prefill chunk.
cases+=(--case "verify8_chunk2k=8*6@$KV:5,2048@$((KV + 2048))c")
cases+=(--case "verify8_chunk8k=8*6@$KV:5,8192@$((KV + 8192))c")
cases+=(--case "chunk8k=8192@$((KV + 8192))c")

torchrun --nproc-per-node="$TP" -m vllm.entrypoints.cli.main bench forward \
    --model "$MODEL" \
    ${REVISION:+--revision "$REVISION"} \
    --load-format "$LOAD_FORMAT" \
    --tokenizer-mode glm5_next \
    --tensor-parallel-size "$TP" \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.6 \
    --language-model-only \
    --speculative-config "$SPEC_CONFIG" \
    --output-json "$OUT/forward.json" \
    "${cases[@]}" > "$OUT/forward.log" 2>&1
echo "DONE rc=$? elapsed_time=${SECONDS}s" | tee "$OUT/status.txt"
