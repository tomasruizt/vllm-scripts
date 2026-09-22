# DFlash / GDN benchmarks

[Open the B200 report](https://tomasruizt.github.io/reports/b200-dflash/).
It compares vLLM, SGLang, their DFlash variants, and vLLM + DFlash PR 52297 on Qwen 4B, 27B, and 35B-A3B at c=1,2,4,8,16,32, for both K=7/8 and K=15/16.

- Recorded releases: vLLM 0.30.0 and SGLang 0.5.20; PR 52297 merged onto vLLM 0.30.0. Mamba convolution and SSM states use BF16.
- vLLM counts proposed tokens; SGLang counts the block including one conditioning token. Thus vLLM K=7 matches SGLang K=8.
- This repository contains source scripts and authored documentation. Generated reports, logs, metric exports, profiler traces, and evidence manifests stay outside Git.
- The existing GitHub Pages report is published separately in `tomasruizt.github.io`. Its Bench links expose metric exports and formulas; those published files are not imported into this repository.
- `notes/B200_*` records the B200 investigations; the other notes and [revisions.json](revisions.json) document the earlier H100 investigation. Historical artifact paths refer to local files, not files included in a fresh clone. `scripts/setup.sh` installs those older H100 dependencies, not the B200 releases.

## Run on B200

Run from this directory with the recorded engine environments and `canhazgpu` configured.
`scripts/run_b200.sh` identifies the environment paths; adapt them for another machine.

```bash
# Arguments: output directory, vLLM proposal count, parallel jobs, starting port.
bash scripts/run_b200_matrix.sh "$PWD/results/b200-block8-20260922" 7 7 8500
bash scripts/run_b200_matrix.sh "$PWD/results/b200-latest-20260922" 15 7 8600
```

Each job reserves one GPU through canhazgpu.
Use a new output directory for reruns; these commands launch benchmarks, not just report generation.

## Render and share

Rendering requires local benchmark outputs; a source-only clone has no measurements.

```bash
~/.venv/bin/python scripts/plot_pareto.py --help
~/.venv/bin/python scripts/combine_b200_results.py \
  --block8 results/b200-block8-20260922 \
  --block16 results/b200-latest-20260922 \
  --output results/RESULTS.html
~/.venv/bin/python scripts/package_b200_report.py --output dist/b200-report-new
```

Generate each model's plots with `plot_pareto.py` before rendering the combined report.
The packaging output directory must be new; its folder works offline or as a static site, and the ZIP includes the linked metric exports.
Keep generated bundles in artifact storage or a separate publishing location, not source commits.
