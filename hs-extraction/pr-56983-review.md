# Review: vLLM PR #56983

Reviewed [PR #56983](https://github.com/vllm-project/vllm/pull/56983) at commit `8a1cf967f87e5ba3cad0c44b5939379fcd5f5cb0`.
The capture logic looks reasonable, but I would request validation changes before approval.

1. **The reported serving results do not validate the current PR.** Only aux capture remains; the KV-cache changes were removed. I reproduced the indexer page-size failure using GLM-like specs plus the draft under `LBHNC`, the only layout supported by the reported SM90 backend. This can land as a prerequisite, but the description should stop claiming standalone end-to-end support and identify the remaining dependency. [Current changes](https://github.com/vllm-project/vllm/pull/56983/files), [SM90 layout restriction](https://github.com/vllm-project/vllm/blob/8a1cf967f87e5ba3cad0c44b5939379fcd5f5cb0/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py#L172-L173).

2. **Add regression tests for the actual changed code.** Cover deferred mHC completion against an independent numerical reference, unchanged target output/state with capture enabled, non-mHC without double-adding residual, and first/intermediate/final capture boundaries. I ran nine targeted CPU checks; all passed, using mocked decoder layers and the PyTorch mHC reference. Those checks should become maintained tests. [Capture implementation](https://github.com/vllm-project/vllm/blob/8a1cf967f87e5ba3cad0c44b5939379fcd5f5cb0/vllm/models/glm5next/common/model.py#L704-L787).

3. **Validate GPU SP and CUDA-graph execution explicitly.** The all-gather is now present, but TP8+EP with DP1 does not exercise `use_sequence_parallel_moe`. My follow-up CPU test passed 12 scenarios over two Gloo ranks, using the real capture loop and SP helpers with decoder fixtures. This verifies gathering, ordering, and padding removal, but does not exercise NCCL or actual attention/MoE execution. Test TP2+DP2+EP and compare aux tensors and target outputs against a reference. Then check acceptance under eager versus full CUDA graphs; GSM8K accuracy alone cannot detect broken draft features. [SP activation condition](https://github.com/vllm-project/vllm/blob/8a1cf967f87e5ba3cad0c44b5939379fcd5f5cb0/vllm/config/parallel.py#L719-L735).

Minor code cleanup: `forward()` still declares `-> torch.Tensor` although it now returns a tuple.

No confirmed capture-math bug was found.
GPU kernels, CUDA graphs, and full distributed model execution remain unverified.
The [failed CI check](https://github.com/vllm-project/vllm/actions/runs/36296895063/job/108557297263) is an eligibility gate; pre-commit was skipped, so it provides no code-validation signal.

The follow-up confirmed that both causal-LM and conditional-generation wrappers configure the inner decoder and forward aux outputs correctly.
The DFlash layer-ID conversion also matches capture-before-layer semantics; no additional code defect was found.
PR #58834 addresses a separate, pre-existing non-mHC decoder SP issue; these fixture-based checks do not validate that decoder fix.

## Local validation

- Capture checks (`test_capture.py`) — 9 passed, using the PyTorch mHC reference and mocked decoder layers.
- Interface and SP checks (`test_interfaces_sp.py`) — 3 tests passed: 2 wrapper checks and a distributed test covering 12 scenarios on two CPU Gloo ranks. SP cases cover 1/5/8 tokens, mHC/non-mHC boundaries, and capture enabled/disabled, compared exactly against unsharded execution. The communicator uses Gloo and decoder layers are fixtures; no real DP/EP model or GPU kernels are exercised.
- Synthetic KV-cache reproduction (`check_cache.py`) — `LBHNC` raises the indexer page-size `NotImplementedError`; `BLHNC` produces 7 groups, but the reported SM90 backend does not support that layout.
- `git diff --check eb0f2ca HEAD` — passed.
- Review checkout: `/home/tomasruizt/code/vllm-review-56983`.

Validation scripts are local artifacts in `/home/tomasruizt/code/pr-56983-validation`, not included in this repository.

```bash
cd /home/tomasruizt/code/pr-56983-validation
export VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1
export PYTHONPATH=/home/tomasruizt/code/vllm-review-56983
/home/tomasruizt/.venv/bin/python -m pytest test_capture.py test_interfaces_sp.py -q --confcutdir=. --rootdir=. -p no:cacheprovider
/home/tomasruizt/.venv/bin/python check_cache.py
```

The distributed test requires local loopback access.

## Proposed GPU validation

Run these in priority order.
The first three directly validate the remaining PR code without the full checkpoint or a KV allocation fix.

| GPUs | Validation | Pass criteria |
|---|---|---|
| 1 GPU | Run real mHC layers with aux capture enabled/disabled. Compare captures against independently materialized `hc_post` + contraction. Cover first/intermediate/final boundaries and varied token counts. | Correct aux values; capture leaves target outputs and carried state unchanged. |
| 1 GPU | Repeat under CUDA-graph capture/replay, changing input values between replays and exercising padded batch sizes. | Eager/graph agreement; no stale aux tensors or memory errors. |
| 4 GPUs: TP2 × DP2, EP enabled | Run small, randomly initialized GLM layers with real attention/MoE and NCCL. Compare SP against an equivalent unsharded-token reference. Include 1/5/8 tokens and different token counts per DP replica. | Correct token ordering, padding removal, aux shapes and numerical agreement; no collective hangs. |
| Enough memory for the full model | Start the exact PR head with GLM + DFlash2 and the advertised configuration. | Establish whether startup still fails at KV allocation. Record any companion patch separately. |
| Full-model setup, once startup works | Compare target-only and DFlash2: eager/full graphs, single/batched requests, short/long prompts, mixed prefill/decode, prefix-cache reuse and preemption. | Comparable accuracy/logprobs, healthy acceptance, and no crashes or state corruption. |

Measure accuracy and acceptance together: rejection can preserve answer quality even when draft features are broken.
Greedy text equality alone is insufficient on the reported nondeterministic stack.
Keep the known non-mHC SP bug separate when interpreting failures.
