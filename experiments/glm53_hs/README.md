# GLM-5.3-Flash hidden-state validation

Scripts and reports for validating vLLM extraction through Speculators. Run evidence is archived in `~/benchmarks/glm-hs-validation/`; generated tensors and the temporary HF cache were removed during cleanup, so auditing old outputs requires regenerating them.

Run `HS_OUTPUT_DIR="$HOME/benchmarks/glm-hs-validation/new-run" bash run-api.sh 1` from this directory. Use a fresh output directory. `run-api-piecewise.sh` selects explicit piecewise graphs. These launchers reserve four GPUs through canhazgpu.

Set `VLLM_DIR` to a checkout containing the extraction changes; it defaults to `~/code/vllm`, whose compatibility is not established by this cleanup. The tested commits and scope are in RESULTS.md and VALIDATION.md. `MODEL_DIR` and `HS_RUNS_DIR` are configurable; the default model uses the original shared `/data` snapshot.

Older diagnostic and same-pass scripts are retained for reproduction. Historical manifests refer to tensors that are no longer present. Archived reports retain their original paths; the reports here point to the archive.
