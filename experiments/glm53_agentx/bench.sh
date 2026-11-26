#!/usr/bin/env bash
# One AgentX point against a running server (agentx-harness, as in InferenceX's docs/agentx-standalone.md).
# Env: CONC, DURATION (s), WARMUP (requests per lane), OUT (result dir).
set -euo pipefail

MODEL=zai-org/GLM-5.3-Flash
URL=http://127.0.0.1:8000
# The agentx preset runs 3600 s. Shorter runs need --unsafe-override (InferenceX: below 900 s).
[[ $DURATION -ge 900 ]] || short=(--unsafe-override)

/models/agentx-venv/bin/agentx profile \
    --url "$URL" --endpoint /v1/chat/completions \
    --model "$MODEL" --tokenizer "$MODEL" --tokenizer-trust-remote-code \
    --concurrency "$CONC" \
    --benchmark-duration "$DURATION" \
    --warmup-requests-per-lane "$WARMUP" \
    --public-dataset semianalysis_cc_traces_weka_062126 \
    --server-metrics "$URL/metrics" \
    --output-artifact-dir "$OUT/aiperf_artifacts" \
    "${short[@]}"
