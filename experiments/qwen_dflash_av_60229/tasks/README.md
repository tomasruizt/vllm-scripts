# Shared setup for the AV follow-up tasks

Each `task-*.md` in this folder is one self-contained task for one agent. Read this file first, then your task file, then `../FINDINGS.md` (established results, superseded claims, runner pitfalls). Claim your beads issue with `bd update <id> --claim` before starting and close it when done.

## Background in three sentences

We benchmarked adaptive verification (AV, vLLM PR #60229) on Qwen3.5-35B-A3B with a DFlash drafter (k=7) on one B300. AV looked 4 to 7% slower at concurrency 128 because every step with a prefill exceeds the 1,024-token CUDA graph capture limit and runs eagerly and CPU-bound (~90 ms of wall time for ~20 ms of GPU work). With extra capture sizes up to 6,144 the AV penalty disappears and both modes run 25 to 37% faster. The human-readable report is `../av-cost-model.html`.

## Environment

- Host: the B300 machine, ssh alias `RH-nm-ore-b300-01` (8 GPUs, shared with other users).
- GPUs: always reserve via canhazgpu, e.g. `canhazgpu run --gpu-ids 4 --timeout 2h --note "<task>" -- <cmd>`. Check `canhazgpu status` first.
- vLLM: `~/code/vllm-validate-60229` is PR #60229 at commit `888ec6bb`, precompiled editable install, used through `PYTHONPATH`. Never modify it. For changes, copy it (`rsync -a --exclude=.git ~/code/vllm-validate-60229/ ~/code/vllm-<task>/`) and set `PYTHONPATH` to the copy.
- Never start a server from inside a vLLM checkout: `python -m` puts the cwd ahead of `PYTHONPATH` and silently imports the wrong code. Check with `python -c "import vllm; print(vllm.__file__)"` using the same env and cwd.
- Python: `~/.venv/bin/python`. Env vars used in all runs: `VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=1024`.
- Target model: `/data/tomasruizt/hub_cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754` (served as `qwen`, `--language-model-only`).
- Drafter: `z-lab/Qwen3.5-35B-A3B-DFlash` at revision `52cb554b4995dede3e2e1bdb129cdb1f3529332b`.
- Server flags of the benchmark: `--max-model-len 16384 --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 --no-enable-prefix-caching --disable-uvicorn-access-log --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-35B-A3B-DFlash","revision":"52cb554b4995dede3e2e1bdb129cdb1f3529332b","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}'`.
- Load generator: `~/aiperf-venv/bin/aiperf` (0.13.0) with `../steady-speedbench.yaml`. Full harness: `../run-steady.sh` (set `PORT_A`/`PORT_B` to ports nobody else uses).
- `vllm bench forward` (PR #61339, merged 2026-10-14) times exact batch shapes as one-shot CUDA graphs. It is not in the PR #60229 checkout. Only if your task needs it, install a recent upstream main into a separate venv with precompiled wheels (`VLLM_USE_PRECOMPILED=1 uv pip install -e .`), never into `~/.venv`. Everything else stays on the PR #60229 checkout.

## Rules

- Follow `~/code/rh/AGENTS.md` (concise reports, standard tooling, async remote jobs, no `cat` of remote files, copy results locally and query with duckdb).
- Record results in `../FINDINGS.md` (append a dated entry, do not delete others) and as a comment on your beads issue (`bd comments add <id> "..."`).
- Report the exact commands you ran, with all flags, and the numbers as bullet points or tables.
- Do not publish anything and do not open upstream PRs or issues. Propose them in your result instead.
