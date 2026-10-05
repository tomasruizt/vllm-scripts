# GLM-5.3 report

From the `vllm-scripts` root:

```bash
~/.venv/bin/python experiments/glm53_av/analysis/build_report.py
```

This rebuilds `experiments/glm53_av/report/` from its archived `data/`. To import completed runs, add `--sources experiments/glm53_av/analysis/sources.json`; that optional file contains local run paths and is ignored by Git. Benchmark notes are in `experiments/glm53_av/glm-5.3-ep-dp-av.md`.

Open the generated `index.html` directly. It contains one throughput/interactivity chart and two results tables in tabs. The CSS comes from `reporting/b200-theme.css`, copied from the [original B200 report](https://tomasruizt.github.io/reports/b200-dflash/). Matplotlib is required in the uv-managed Python environment.

Analysis code and documentation belong in Git; raw inputs and generated reports do not. Keep the output directory locally: `data/` holds the archived inputs and launch provenance, `plots/` holds PNG/SVG exports, and `results.csv`/`results.json` hold derived metrics. `data-sha256.json` records input hashes.

Agent workflow and metric rules are in [AGENTS.md](AGENTS.md) and [the shared reporting instructions](../../../reporting/AGENTS.md).
