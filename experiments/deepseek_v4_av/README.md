# DeepSeek V4 DEP4 padding reclamation

Compare AV on against AV on + padding reclamation using AIPerf 0.13.0. Ten runs per method at concurrency 8, 16, 32, 64, 128 and 256. Use the official DeepSeek-V4-Flash-DSpark checkpoint at revision `62af8fffb2f7030cac4de2f0169f5b8d1101b646`, TP1/DP4/EP4, on four B300s.

NVIDIA's preparation script resolves SPEED-Bench throughput-1K into 1,536 single-turn prompts across three entropy tiers. This performance workload forces 512 output tokens per request (`ignore_eos`, `min_tokens`), rather than measuring natural answer lengths. It does not measure accuracy.

Reset prefix cache before every measured run; keep prefix caching enabled within each run. Save AIPerf request records, summaries, commands, Prometheus snapshots, cache-reset acknowledgements and server/client logs under `/data/tomasruizt/benchmarks/deepseek-v4-dep4-av/`. Keep inputs and generated artifacts out of Git. Reserve GPUs with `chg run`.

`benchmark.py` reuses the GLM runner's server health, cache reset and metric helpers. It fails on request errors, missing speculative activity, unexpected output length or failure to exercise reclamation. It uses one server per method and excludes startup and warmup from the measured runs.

Build the local report from the current GLM reference:

```bash
~/.venv/bin/python experiments/deepseek_v4_av/build_report.py \
  --glm-report experiments/glm53_av/data/report-padding-review \
  --sources /tmp/deepseek-report-sources.json
```

The optional sources file maps `avon` and `reclaim` to completed run directories. It imports the summaries, cache-reset responses, metric snapshots, commands, and client/server logs into the report's ignored `deepseek-v4/logs/` directory. Subsequent builds omit `--sources`; the archived inputs are sufficient. The generator reuses the GLM renderer, template, theme, and plotting helpers, then adds model tabs with `reporting/model_tabs.py`.

GLM stays selected initially. DeepSeek has throughput, TPOT, and AL tabs; it has no accuracy measurements. All ten repeats remain in the plot and tables. The report states that the slower first three baseline repeats confound the apparent performance gain. Plot individual runs in each method's color at alpha 0.7, with x = 1000 / that run's mean TPOT in milliseconds. Curves connect aggregate means; tables show means. Keep SD in exported data, but do not draw SD bars.

Always rebuild the combined report after rebuilding GLM alone. Check both model tabs and all table tabs in a browser; verify unique HTML/SVG IDs, rendered table values, readable title colors, local log links, and archive hashes. See [shared reporting instructions](../../reporting/AGENTS.md). Building is local and does not publish anything.

For repeatable concurrency sweeps, use `short_sweep.py` rather than sending the full dataset at every point. It estimates prompt counts from completed reference runs, performs one live calibration per concurrency, and freezes those counts for both methods. Each repeat uses a seeded, shuffled subset balanced across the three entropy tiers; both methods use the exact same subset and order. It targets 30 seconds of inference, not 30 seconds including client startup.

The method order is half the baseline repeats, all reclamation repeats, then the remaining baseline repeats. Startup and calibration are excluded, prefix cache is cleared before every measurement, and completed points are resumable. Measurement subprocesses have a three-minute minimum timeout rather than waiting hours on a stalled client. Queue a persistent `nohup setsid chg run` launcher so the reservation heartbeat survives the tool session. Run `--prepare-only` against a temporary output directory to validate estimated subsets without GPUs; calibration can change their sizes, so do not reuse that directory for the real sweep.
