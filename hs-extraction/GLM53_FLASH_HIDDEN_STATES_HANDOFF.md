# GLM-5.3-Flash Batched Hidden-State Extraction Handoff

## Executive summary

We added selectable auxiliary hidden-state extraction for `zai-org/GLM-5.3-Flash` and validated it with concurrent, mixed-length requests on GB300 hardware.

The model-side capture logic works. The major batching failure was not caused by the `ExampleHiddenStatesConnector`: GLM's custom KV-cache layout gave the native MLA cache and the hidden-state cache overlapping physical storage even though they used independent block tables. With concurrent requests, the same numeric block ID could belong to different requests in the two groups, allowing a hidden-state write to corrupt another request's live attention cache.

The validated fix reserves disjoint physical storage for the hidden-state cache and includes that storage in cache-memory accounting.

Validated vLLM branch:

- Fork: `https://github.com/shanjiaz/vllm`
- Branch: `glm5-next-aux-hidden-state-concurrency-fix`
- Model integration commit: `9959fabf9ced004a32a26c5d91138db235ec11be`
- Cache non-aliasing fix: `7e1d083ce66d9e85f783252bf524a5de9ef85bb6`
- PR creation link: `https://github.com/shanjiaz/vllm/pull/new/glm5-next-aux-hidden-state-concurrency-fix`
- Runtime reported by the validated server: `v0.28.1rc1.dev580+g385dce36b`

Speculators checkout:

- Path: `/mnt/shared/home/shanjiaz/speculators-pr996-glm53-flash-layer-search`
- Commit: `b9c51bcfc134923eae2b6a6f8246ae3500d5c42f`
- Based on Speculators PR `#996`

## Relevant paths

| Purpose | Path |
|---|---|
| Fixed vLLM worktree | `/mnt/shared/home/shanjiaz/vllm-glm5-aux-concurrency-fix` |
| Earlier integration worktree | `/mnt/shared/home/shanjiaz/vllm-glm5-eagle3-pr` |
| Speculators worktree | `/mnt/shared/home/shanjiaz/speculators-pr996-glm53-flash-layer-search` |
| Model cache | `/data/shanjiaz/hf/hub/models--zai-org--GLM-5.3-Flash` |
| Model snapshot | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` |
| Valid server launcher | `/mnt/shared/home/shanjiaz/run_glm53_flash_aux_smoke.sh` |
| Concurrent stress test | `/mnt/shared/home/shanjiaz/stress_glm53_flash_hidden_states_summary.py` |
| Single-sample diagnostic | `/mnt/shared/home/shanjiaz/diagnose_glm53_flash_hidden_states.py` |
| Valid layer-search output | `/data/shanjiaz/glm53-flash-layer-search/coarse-20260922-concurrency-fixed` |

## Model and layer semantics

GLM-5.3-Flash has 45 transformer layers. Its repeating pattern is approximately:

```text
3 linear-attention layers -> 1 full-attention/MLA layer
```

The extraction IDs are **layer boundaries**, not zero-based layer module IDs:

- Boundary `20` means the representation after transformer layer `19`.
- Full-attention layers are `3, 7, 11, ..., 43`.
- Corresponding post-full-attention boundaries are `4, 8, 12, ..., 44`.
- Boundary `45` is the final model boundary and is included implicitly by the training/extraction setup.

GLM-5.3-Flash uses mHC (Manifold-Constrained Hyper-Connections). A raw internal `hidden_states` tensor can be an incomplete representation because the layer also carries deferred `residual`, `post`, and `comb` state. For an mHC boundary, extraction applies:

```python
hidden_states = layer.hc_post(hidden_states, residual, post, comb)
hidden_states = hc_contract(hidden_states, layer.n)
```

This yields the normal model-width representation suitable for training the speculator.

## Changes made in vLLM

### 1. GLM5-Next boundary capture

File:

```text
vllm/models/glm5next/common/model.py
```

The integration:

- accepts selectable `eagle_aux_hidden_state_layer_ids`;
- captures layer-boundary representations;
- handles the embedding boundary when requested;
- materializes the complete mHC boundary with `hc_post` and `hc_contract`;
- returns the final boundary when requested;
- gathers sequence-parallel shards before returning auxiliary tensors.

### 2. GLM hybrid-cache recognition

File:

```text
vllm/v1/core/kv_cache_utils.py
```

GLM has a custom hybrid allocation containing Mamba/linear-attention state, MLA state, sparse-indexer state, and optional k-pool tail state. `HiddenStateCacheSpec` must be excluded from native MLA classification and represented as its own cache group.

### 3. Critical concurrency fix: disjoint physical storage

File:

```text
vllm/v1/core/kv_cache_utils.py
```

The original integration appended a hidden-state cache group but assigned its tensor offset to byte `0` and used `max(native_bytes, hidden_bytes)` for per-block allocation. This overlapped native and hidden-state storage.

The fix:

```python
hidden_bytes = sum(
    group.kv_cache_spec.page_size_bytes
    for group in kv_cache_groups
    if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
)
bytes_per_block = native_bytes + hidden_bytes
```

Hidden-state tensors are then placed after all native MLA/indexer pages:

```python
hidden_offset = (
    len(mla_names) * mla_page + len(idx_names) * idx_page
) * num_blocks
```

Each hidden-state group advances this offset by its allocated size. A regression test verifies that the hidden tensor begins after every native tensor region.

### 4. Block-zeroing compatibility

File:

```text
vllm/v1/worker/utils.py
```

`HiddenStateCacheSpec` is excluded from the attention-cache zeroer. It is connector-owned storage, fully overwritten for produced tokens, and is not consumed by attention kernels.

### 5. Kernel-warmup compatibility

File:

```text
vllm/model_executor/warmup/kernel_warmup.py
```

The integration contains a small warmup compatibility adjustment required by this model/runtime combination. Review commit `9959fabf9c` for the exact diff.

## Connector behavior and required timeout

The connector performs an asynchronous GPU-to-CPU copy followed by a safetensors disk write. The response contains a unique handle such as:

```text
/out/hidden_states/cmpl-...safetensors
```

While the file is being produced, a sibling `.lock` file remains locked. The training-side connector originally waited only 10 seconds. Large concurrent tensors can legitimately take longer, producing a false failure even though the server returns HTTP 200 and eventually completes the file.

We made the wait configurable in:

```text
hs_connectors/src/hs_connectors/transfer.py
```

and launch training with:

```bash
export HS_CONNECTOR_LOCK_TIMEOUT=300
```

This is an I/O completion timeout, not a request timeout. Keep the ordinary OpenAI/vLLM request timeout at 300 seconds as well.

## How to launch vLLM hidden states extraction

The commands below are intentionally machine-independent. Replace only the
paths and GPU count for the colleague's environment.

### 1. Check out the required sources

```bash
git clone https://github.com/shanjiaz/vllm.git
cd vllm
git checkout 7e1d083ce66d9e85f783252bf524a5de9ef85bb6
git rev-parse HEAD
# Expected: 7e1d083ce66d9e85f783252bf524a5de9ef85bb6

cd ..
git clone https://github.com/vllm-project/speculators.git
cd speculators
git fetch origin pull/996/head:pr-996
git checkout b9c51bcfc134923eae2b6a6f8246ae3500d5c42f
git rev-parse HEAD
# Expected: b9c51bcfc134923eae2b6a6f8246ae3500d5c42f
```

Install the patched vLLM checkout in the environment that will run the server.
Use the dependency-installation method appropriate for that machine; the
important point is that Python imports this checkout, not an unrelated PyPI
vLLM installation. Verify it before launching:

```bash
python -c 'import vllm; print(vllm.__file__); print(vllm.__version__)'
```

### 2. Launch hidden-state extraction directly

Set these four paths for the local machine:

```bash
export VLLM_REPO=/path/to/vllm
export SPECULATORS_REPO=/path/to/speculators
export MODEL=zai-org/GLM-5.3-Flash
export HIDDEN_STATES_DIR=/path/to/writable/hidden_states

mkdir -p "$HIDDEN_STATES_DIR"
```

Expose the Speculators plugin and launch the server:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_PLUGINS=dynamic_hidden_states
export VLLM_DEEP_GEMM_WARMUP=skip
export PYTHONPATH="$SPECULATORS_REPO/vllm_plugins:$SPECULATORS_REPO${PYTHONPATH:+:$PYTHONPATH}"

python "$SPECULATORS_REPO/scripts/launch_vllm.py" "$MODEL" \
  --hidden-states-path "$HIDDEN_STATES_DIR" \
  --target-layer-ids 20 28 32 36 40 44 \
  --port 8010 \
  --tensor-parallel-size 4 \
  --max-model-len 16385 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 32 \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --kernel-config '{"enable_jit_warmup":false,"enable_cutedsl_warmup":false,"enable_flashinfer_autotune":false}' \
  --enforce-eager \
  --worker-extension-cls dynamic_hidden_states.worker_extension.AuxLayerWorkerExtension
```

This runs in the foreground and prints the actual startup failure if anything
is misconfigured. Only after it starts successfully should it be moved under
the colleague's preferred process supervisor (`systemd`, Slurm, Kubernetes,
`screen`, etc.). No particular supervisor is required for extraction.

For eight GPUs, change both settings consistently:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# ...use the same launch command, but replace:
# --tensor-parallel-size 4
# with:
# --tensor-parallel-size 8
```

The six values supplied to `--target-layer-ids` are the explicit auxiliary
boundaries. Final boundary `45` is added by the extraction/training protocol.

### 3. Verify the running server

```bash
curl -f http://127.0.0.1:8010/health
curl -s http://127.0.0.1:8010/v1/models | python -m json.tool
curl -s http://127.0.0.1:8010/aux_hidden_state_layers
```

Change the selected boundaries without restarting:

```bash
curl -f -X POST \
  -H 'Content-Type: application/json' \
  -d '{"layers":[20,28,32,36,40,44,45]}' \
  http://127.0.0.1:8010/aux_hidden_state_layers
```

The directory passed through `--hidden-states-path` must be writable by the
server and readable by the extraction client/training workers. On multiple
nodes, use a genuinely shared filesystem and provide the same logical path to
both sides.

Why these flags:

- `--enforce-eager`: the PR #996 compiled masked-accumulate path did not cover the GLM5-Next implementation during this work. Eager mode was the validated path.
- `--no-enable-prefix-caching`: changing captured layer sets while retaining cached prefixes previously produced stale/mismatched captures. Keep it disabled for layer-selection work.
- `--max-num-batched-tokens 32768` and `--max-num-seqs 32`: validated concurrent settings, not strict maxima.
- `--max-model-len 16385`: permits an exact 16,384-token training sample plus the one generated token used to trigger extraction.
- `--kv-cache-dtype fp8`: matches the intended GLM-5.3-Flash serving setup.

## Minimal extraction request

The prompt may be text or token IDs. Token IDs avoid chat-template differences.

```python
import os
from pathlib import Path

import openai
from safetensors.torch import load_file

endpoint = "http://127.0.0.1:8010"
shared_root = Path(os.environ["HIDDEN_STATES_DIR"])

client = openai.OpenAI(base_url=f"{endpoint}/v1", api_key="EMPTY")
model = client.models.list().data[0].id
response = client.completions.create(
    model=model,
    prompt=[1, 2, 3, 4],
    max_tokens=1,
    temperature=0,
    extra_body={"return_token_ids": True},
    timeout=300,
)

handle = response.kv_transfer_params["hidden_states_path"]
path = shared_root / Path(handle).name
# In production, wait on the sibling `.lock` with flock before reading.
tensors = load_file(str(path))
print(tensors["token_ids"].shape)
print(tensors["hidden_states"].shape)
```

Expected hidden-state shape:

```text
[prompt_tokens, requested_boundaries_including_final, hidden_size]
```

For six explicit boundaries plus final boundary 45 on GLM-5.3-Flash:

```text
[prompt_tokens, 7, 4096]
```

## Validated concurrent stress test

Run from a machine that can see both the server endpoint and its shared output directory:

```bash
python /mnt/shared/home/shanjiaz/stress_glm53_flash_hidden_states_summary.py \
  http://pod4-gb300-3-tray18-f3:8010 \
  /mnt/shared/home/shanjiaz/glm53-flash-layer-search/native-aux-pod4-gb300-3-tray18-f3 \
  /data/shanjiaz/glm53-flash-layer-search/coarse-data-v2
```

The harness submits 32 concurrent requests using eight prompt lengths repeated four times. Lengths cover approximately 2.7K to 9.2K tokens and include the original 9,172-token reproducer.

Validated result across three rounds:

```text
96/96 requests completed
0 exceptions
0 nonfinite tensors
0 NaNs/Infs
```

Repeated outputs are not bit-identical, even when the same request is executed serially. The FP8/MoE execution showed small run-to-run numerical variation. Therefore, do not use byte equality as the correctness criterion. Check:

1. exact returned token IDs;
2. expected tensor shape;
3. all values finite;
4. no cross-request handle reuse;
5. reasonable numerical agreement with a serialized reference.

## What failed and what it taught us

### Normal batched extraction before cache isolation

Symptoms:

- deterministic or intermittent NaNs;
- the 9,172-token reproducer first showed invalid values around token 2,176;
- failures increased with multiple outstanding requests;
- layer-selection runs could finish with `val/loss=nan` and meaningless acceptance metrics if errors were merely skipped.

Cause:

- hidden-state and native caches physically overlapped while their block tables were allocated independently.

Status:

- fixed by commit `7e1d083ce6`.

### Restricting the server to one active sequence

Tried:

```text
--max-num-batched-tokens 16385
--max-num-seqs 1
```

An isolated long request became finite, but several queued client requests could still reproduce corruption before the physical cache fix. This was a useful safety workaround, not a real solution.

### Serializing the complete client lifecycle

We temporarily serialized request generation, file completion, validation, and deletion across FSDP ranks with a shared `flock`.

This produced valid layer-selection results but destroyed extraction concurrency. It is no longer necessary after the cache-isolation fix and should not be treated as the production design.

### Cache page-alignment-only experiment

We tested a page-alignment adjustment without giving hidden states disjoint physical storage. It did not fix the concurrent stress test and was reverted. Alignment alone was not the issue.

### Prefix caching

A cold-versus-warm layer-swap diagnostic initially appeared to show a final-boundary mismatch. The mismatch came from prefix reuse across changed capture configurations. With prefix caching disabled, warm/cold capture behavior was consistent.

Do not enable prefix caching while dynamically changing auxiliary layer IDs unless cache invalidation is explicitly implemented and tested.

### Ten-second connector lock timeout

After cache isolation, the first real concurrent training attempt failed on a `.safetensors.lock` timeout. The vLLM server had returned HTTP 200 and showed no NaN or GPU failure; the background copy/write simply took longer than the connector's hardcoded 10 seconds.

Status:

- made configurable;
- validated with `HS_CONNECTOR_LOCK_TIMEOUT=300`;
- this change should be upstreamed separately in Speculators/`hs_connectors`.

## Fail-fast requirements

Training must not silently skip invalid extraction samples. The layer-selection launcher used:

```bash
export SPECULATORS_HS_FAIL_FAST=1
export HS_CONNECTOR_LOCK_TIMEOUT=300
```

Before admitting a generated tensor to training, validate:

```python
assert returned_token_ids == requested_token_ids
assert hidden_states.shape == (len(requested_token_ids), 7, 4096)
assert torch.isfinite(hidden_states).all()
```

Also treat these as fatal:

- missing or unreadable output handles;
- a lock that exceeds the extended timeout;
- duplicate request handles;
- a server-side traceback, OOM, CUDA, NCCL, or Xid event.

## Layer-selection result

All candidates used identical initialization, data, optimizer settings, and the fixed concurrent extraction path.

| Candidate | Six explicit boundaries | Validation EAL | Acceptance rate | P0 | P7 |
|---|---|---:|---:|---:|---:|
| Uniform full-attention | `[4,12,20,28,36,44]` | 1.180 | 11.7% | 17.5% | 15.0% |
| Offset | `[6,14,22,30,38,44]` | 1.180 | 11.7% | 17.6% | 15.0% |
| Deep full-attention | `[20,28,32,36,40,44]` | 1.182 | 11.8% | 17.7% | 14.9% |

Boundary `45` was implicit in all cases.

The differences are extremely small. Deep full-attention is the refinement leader, not a conclusive winner. Do not choose a production layer set from these coarse values alone; run matched refinement/repeats and end-to-end acceptance/throughput evaluation.

Leading coarse checkpoint:

```text
/data/shanjiaz/glm53-flash-layer-search/coarse-20260922-concurrency-fixed/deep_full/checkpoints/0
```

Important hashes:

```text
model.safetensors  31b54981d7a104039c19c585eda0d1c593f721c4ba9dcb9823e2b891d2f3c65b
config.json        0b28aade26a5fea70c6593ec890a3b5545714529efe3a29a2135addd75c18cd2
val_metrics.json   8d8ad0ae983683d123fa634710ffacca10ea0bd0920ad98ccc9231187b857f56e
```

## Recommended upstream split

Keep the review surface understandable by splitting the work:

1. **vLLM model integration PR**
   - selectable GLM5-Next auxiliary boundary capture;
   - correct mHC materialization;
   - sequence-parallel gather;
   - worker-extension control endpoint.

2. **vLLM cache-layout bugfix PR or clearly separated commit**
   - `HiddenStateCacheSpec` as its own GLM cache group;
   - disjoint physical allocation;
   - correct per-block memory accounting;
   - regression test proving no overlap.

3. **Speculators/hs-connectors PR**
   - configurable lock timeout;
   - fail-fast option or strict validation policy;
   - ideally expose asynchronous transfer completion more directly than polling a disk lock.

## Remaining work

- Run the focused layer refinement and repeat top candidates with multiple seeds.
- Run end-to-end DSpark acceptance and throughput evaluation; do not select from training loss alone.
- Add a proper integration test that submits concurrent requests through vLLM and validates request-local tensor contents.
- Decide whether dynamic layer changes should invalidate prefix-cache entries or remain incompatible with prefix caching.
- Benchmark concurrency beyond 32 sequences and at the intended 16K prompt distribution.
- Consider replacing filesystem transfer with a faster shared-memory or object-store path for production-scale training.
