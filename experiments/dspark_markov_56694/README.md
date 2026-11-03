# DSpark Markov candidate validation

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
