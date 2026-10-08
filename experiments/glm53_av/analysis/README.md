# GLM-5.3 report

From the `vllm-scripts` root:

```bash
~/.venv/bin/python experiments/glm53_av/analysis/build_report.py
```

This rebuilds `experiments/glm53_av/report/` from its archived `data/`. To import completed runs, add `--sources experiments/glm53_av/analysis/sources.json`; that optional file contains local run paths and is ignored by Git. Benchmark notes are in `experiments/glm53_av/glm-5.3-ep-dp-av.md`.

Open the generated `index.html` directly. It contains one throughput/interactivity chart and two results tables in tabs. The CSS comes from `reporting/b200-theme.css`, copied from the [original B200 report](https://tomasruizt.github.io/reports/b200-dflash/). Matplotlib is required in the uv-managed Python environment.

Analysis code and documentation belong in Git; raw inputs and generated reports do not. Keep the output directory locally: `data/` holds the archived inputs and launch provenance, `plots/` holds PNG/SVG exports, and `results.csv`/`results.json` hold derived metrics. `data-sha256.json` records input hashes.

Agent workflow and metric rules are in [AGENTS.md](AGENTS.md) and [the shared reporting instructions](../../../reporting/AGENTS.md).

To import benchmark and server logs, use a local JSON file mapping session names to completed run directories:

```bash
~/.venv/bin/python experiments/glm53_av/analysis/build_logs.py --sources /tmp/glm53-log-sources.json
```

Each directory must contain `result.json`, `server.log`, `run.py`, and the client logs. The exporter copies logs, results, scripts, and metric snapshots into `report/logs/`, then creates `benchmark-logs.html`, `benchmark-logs.tar.gz`, and `logs/sha256.json`. The no-prefix-cache c=64 repeats are labelled supplementary and stay outside the main plot.

Once logs are imported, the normal `build_report.py` command refreshes the log index and archive too. Neither rebuilding step needs the original run directories. To refresh only the logs, run `build_logs.py` without `--sources`. Both commands accept `--output` for another report directory.

Publish the entire generated report directory to both `reports/glm-5.3-dep4/` and `docs/reports/glm-5.3-dep4/` in the website repository. Its ignore rules exclude `.log` files; explicitly include them:

```bash
git add -f -- ':(glob)reports/glm-5.3-dep4/logs/**/*.log' ':(glob)docs/reports/glm-5.3-dep4/logs/**/*.log'
```

Before publishing, check all relative links and archive checksums. After deployment, fetch the public files and compare them with `logs/sha256.json`; a successful Git push alone does not verify that the logs are available. Keep generated files and local source mappings ignored in `vllm-scripts`.

To add DeepSeek as a model tab in the current repeated-run report, use [the DeepSeek generator](../../deepseek_v4_av/README.md). It requires an explicit `--glm-report` path and reuses the existing GLM template and renderer. Run it again after rebuilding GLM alone, which renders a standalone GLM page.
