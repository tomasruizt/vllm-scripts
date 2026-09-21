# DFlash / GDN investigation

Portable handoff for [vLLM #49730](https://github.com/vllm-project/vllm/issues/49730). Start with [RESULTS.md](RESULTS.md); exact source/model revisions are in [revisions.json](revisions.json).

## What we learned

- Repeated GDN metadata preparation contributes to the vLLM/SGLang gap. Larger attention groups reduce overhead but waste cache capacity; [group-size findings](notes/GROUP_SIZE_FINDINGS.md).
- Our [PR #57962](https://github.com/vllm-project/vllm/pull/57962) combines a Triton state-index gather with pure-speculative metadata reuse. It helps but does not close the gap.
- [PR2 #52297](https://github.com/vllm-project/vllm/pull/52297) shares metadata for mixed batches too. At 4B c=8 it reached 2,841.7 tok/s versus our PR's 2,502.7; p99 ITL fell from 61.99 to 47.66 ms.
- Median ITL hid vLLM's long gaps: at 4B c=8, 12.5% of our PR's chunk gaps exceeded 50 ms and accounted for 50.2% of summed gap time. Their cause needs a c=8 trace.
- Original SGLang settings severely limited active requests, especially 27B (3). BF16 states, 95% memory and no prefix cache raised the tested caps to 32 (4B) and 16 (27B). Updated engine comparisons still mix configurations; see the table notes.
- ReplaySSM is available in SGLang but was not enabled. vLLM's Qwen GDN speculative implementation remains an open PR. Shorter draft lengths are another experiment; keep equivalent lengths in both engines.

## Resume on another machine

Requires an H100-class CUDA 13 environment, `uv`, and configured `canhazgpu`. Use one source checkout/environment per vLLM revision. PR2's local merge is preserved on `tomasruizt/vllm`, branch `bench/pr52297`; no local worktree is required.

```bash
git clone --branch bench/pr52297 https://github.com/tomasruizt/vllm.git ../vllm-pr2
git -C ../vllm-pr2 checkout --detach d31939925f5f98ad86604266abb3b2b14a94b663
# Run from this experiment directory:
bash scripts/setup.sh vllm .deps/vllm ../vllm-pr2
ENGINE_PYTHON="$PWD/.deps/vllm/bin/python" bash scripts/run.sh vllm dflash \
  --vllm-dir ../vllm-pr2 --model-size 4B --concurrencies 1 7 16 32 \
  --port 8804 --output "$PWD/results/new/pr2/4B"
```

Repeat with `--model-size 27B`, a distinct port and output directory. These are the outstanding PR2 runs; they use 100/10 requests/warmups at c=1 and 20c/2c otherwise. Prefix caching is disabled in the current launcher for both engines. It starts each server once and reuses it for the sweep.

```bash
bash scripts/setup.sh sglang .deps/sglang
ENGINE_PYTHON="$PWD/.deps/sglang/bin/python" bash scripts/run.sh sglang dflash \
  --model-size 27B --sglang-max-running-requests 16 --concurrencies 1 8 16 32 \
  --port 8727 --output "$PWD/results/new/sglang/27B"
```

The setup recipe preserves recorded dependencies but has not been installed/tested on the destination machine. SGLang intentionally overrides its FlashInfer/CUTLASS metadata pins. Model sampling defaults and EOS stopping remain enabled. No weights or environments are included.

## Contents and preservation

- `snapshots/run.py` is the exact uncommitted runner copied from the experiment checkout; `run.py.patch` preserves its diff against commit `d8a994a636`. `scripts/run.py` is the portable continuation, with an explicit `--vllm-dir` option. Original files remain untouched.
- `results/` contains successful and failed runs, AIPerf metrics, configs, logs, and historical tables; `results/original/` and `results/legacy-raw/` preserve the early baseline/DFlash comparisons. Partial PR2 startup attempts contain no throughput results. Old scripts embedded in results/snapshots are historical; use `scripts/` to resume.
- `profiles/` contains both engines' PyTorch traces and Nsight SQLite timelines. The newer Triton profile is in `results/nsys-triton-20/`. Raw `.nsys-rep` files are excluded because opaque binaries cannot be fully sanitized; originals remain on the source machine. SQLite exports preserve the GPU timing evidence.
- `notes/` preserves detailed analyses and earlier experiments; some references describe the historical layout. `artifact_manifest.json` records source and sanitized content hashes for imported artifacts.
- Copied text, compressed traces and SQLite strings have machine paths, hostnames and recognizable credentials redacted. SQLite environment tables are cleared and databases vacuumed. No environment files, credentials, model weights, caches or raw Nsight binaries are published.
- The original machine had PR2 4B/27B runs queued when this snapshot was taken. Queue state is machine-local and does not migrate; cancel any surviving jobs there before duplicating them elsewhere.

Next: finish PR2 at c=1,7,16,32; rerun main/our PR with prefix caching disabled for matched comparisons; profile the c=8 long gaps. Do not combine c=7 and c=8 measurements into one row.
