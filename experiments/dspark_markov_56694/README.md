# DSpark Markov candidate validation

## Fused revision (November 6, 2026)

Validate the author's November 5 claim that the restored fused Markov walk makes top-k profitable, at `0e5254884c`. This run is single-GPU, unlike the author's DP8 measurement.

Three arms, all with the same checkpoint and server flags:

- `dense`: full-vocabulary DSpark.
- `union32`: unmodified PR source with `dspark_draft_topk=32`, which reaches the fused greedy kernel (`fusion-probe/fusion.txt` confirms it is active).
- `union32_unfused`: the same tree with `can_fuse_markov_walk` forced to `False`, the author's fusion ablation.

Each mode runs on a fresh server per round, with the mode order rotated across four rounds (c=4 was repeated for three more rounds, 57 cells total). Cells send a fixed request count (`16/64/96/256` at c=`1/4/8/32`) instead of a fixed duration, so every arm processes the same prompt prefix under the author's workload policy (temperature 0, natural EOS, 3072-token cap). Whole-run output throughput is therefore matched work, unlike a duration-bounded window. Host CPU load is sampled every 5 s and gated at 48 of 160 cores.

```bash
cd experiments/dspark_markov_56694
export HF_HUB_CACHE=/path/to/writable/shared/cache
export ROOT=/path/to/fresh/run
bash setup-fused.sh
nohup chg run --gpu-ids 5 --timeout 4h --note 'DSpark 56694 fixed matrix' -- \
  env ROOT="$ROOT" ROUNDS=4 WORKLOADS='8k' bash run-fused.sh > "$ROOT/full.log" 2>&1 &
```

`setup-fused.sh` installs the per-commit wheel of the PR merge base (`1cb31be7`) and overlays the three PR Python files, so the compiled extensions match the base exactly and the Triton walk is JIT-compiled from the PR source. Analyze with `analyze.py --self-saturating` plus `compare-fused.py`, which reports per-round paired ratios and an exact sign-flip test.

## Earlier revision (November 2, 2026)

Validate vLLM PR #56694 at `f94d7ff8ae9a` (head tested November 2, 2026). Compare full-vocabulary DSpark, upstream base-logit top-32 pruning, and the PR's base/Markov candidate union using Qwen3-4B and the author's DSpark checkpoint, K=8.

On a Linux GPU host, use an existing compatible precompiled vLLM checkout. Setup downloads pinned Python sources, attaches the same runtime binaries to both controls, installs AIPerf 0.13.0, and reconstructs SPEED-Bench using NVIDIA's pinned loader. It requires a fresh run directory and a writable shared model cache.

```bash
cd experiments/dspark_markov_56694
export RUNTIME=/path/to/precompiled-vllm
export HF_HUB_CACHE=/path/to/writable/shared/cache
export ROOT="$PWD/results/$(date +%Y%m%d-%H%M%S)"
bash setup.sh
nohup chg run --gpus 1 --timeout 2h --note 'DSpark Markov validation' -- bash "$ROOT/run.sh" > "$ROOT/launch.log" 2>&1 &
```

`sweep.yaml` defines c=1/8, three 30-second trials, natural EOS with output cap 1024, and thinking disabled. For c=4, use `[4]` in a separate fresh run directory and `MODES='dense union32'`. Verify GPU clocks and temperature before including results. The historical GPU5 run and c=64 pilots remain excluded by the collector. Preserve failed runs and use a new `ROOT` for retries.

The runner saves the unchanged targeted test-file results separately from performance. The recorded 18-test diagnostic initialized `draft_logits_num_shards=1` and `draft_logits_shard_idx=0` in the cache-replay test fixture. Production PR code was unchanged. Greedy response comparisons are diagnostic, default online serving is not batch invariant.

Fetch stable exports with `make fetch-results REMOTE=<benchmark-host> RUN_ID=<run-id>`, overriding `REMOTE_ROOT` if needed. `prepare_data.py` only adapts NVIDIA's resolved prompts to AIPerf's message schema. `check_outputs.py` saves the eight-request output check. Generated sources, data, reports and exports stay ignored.

Rebuild the recorded comparison locally from the vllm-scripts root:

```bash
EXP=experiments/dspark_markov_56694
RUN=$EXP/results/dspark-markov-56694-20261102-r2
~/.venv/bin/python "$EXP/analyze.py" "$RUN" --extra-source "$EXP/results/dspark-markov-56694-20261102-c4" --extra-source "$EXP/results/dspark-markov-56694-20261102-c4-r2"
~/.venv/bin/python "$EXP/audit_acceptance.py"
(cd "$RUN" && duckdb -csv -f ../../summary.sql > comparison.csv)
```

The comparison contains 48 included trials and 1,906 completed requests. AL is audited from whole-cell server counters, while steady throughput uses the shared window estimator. The small speed changes are exploratory, modes were not interleaved across independent startups and only 15–17 distinct prompts support the matched c=1 AL comparison. Reports and the published comment copy remain under ignored `report/`.
