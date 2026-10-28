REMOTE ?= RH-nm-ore-b300-01
RUN_ID ?= gdn-dspark-block16-20261027-1203-tp2
REMOTE_DIR := /home/tomasruizt/runs/$(RUN_ID)
LOCAL_PYTHON ?= $(HOME)/.venv/bin/python
ANALYZE_FLAGS ?=
PLOT_TITLE ?= Qwen3.6-27B · 2× B300, TP2

.PHONY: status fetch-results analyze report plot-pareto
plot-pareto:
	$(LOCAL_PYTHON) plot.py results/$(RUN_ID)/metrics.csv --output plots/$(RUN_ID)/throughput-inverse-tpot --title '$(PLOT_TITLE)'

report: analyze
	duckdb -c "COPY (WITH a AS (SELECT mode, concurrency, median(throughput) AS tok_s, median(mean_tpot_ms) AS tpot_ms FROM read_parquet('results/$(RUN_ID)/metrics.parquet') WHERE valid GROUP BY ALL) SELECT f.concurrency, f.tok_s AS fixed_tok_s, a.tok_s AS adaptive_tok_s, 100 * (a.tok_s / f.tok_s - 1) AS gain_pct, f.tpot_ms AS fixed_tpot_ms, a.tpot_ms AS adaptive_tpot_ms FROM a f JOIN a USING (concurrency) WHERE f.mode='fixed' AND a.mode='adaptive' ORDER BY concurrency) TO 'results/$(RUN_ID)/comparison.json' (FORMAT JSON, ARRAY true);"
	duckdb -c "COPY (SELECT mode, concurrency, throughput, 1000 / mean_tpot_ms AS interactivity FROM read_parquet('results/$(RUN_ID)/metrics.parquet') WHERE valid) TO 'results/$(RUN_ID)/plot-runs.csv' (HEADER);"
	$(LOCAL_PYTHON) ../qwen_dflash_av_60229/build_report.py --plot-csv results/$(RUN_ID)/plot-runs.csv --output plots/$(RUN_ID) --title 'Qwen3.6-27B + DSpark, K=15' --series 'fixed:Fixed K=15:#64748b:o' --series 'adaptive:Adaptive K=15:#138b83:D'
	$(LOCAL_PYTHON) report.py results/$(RUN_ID) --output report/$(RUN_ID)

analyze:
	$(LOCAL_PYTHON) analyze.py results/$(RUN_ID) --complete $(ANALYZE_FLAGS)
	duckdb -c "SELECT mode, concurrency, count(*) AS runs, round(median(throughput)) AS tok_s, round(median(mean_tpot_ms), 2) AS tpot_ms, round(median(acceptance_length), 3) AS acceptance_length FROM read_parquet('results/$(RUN_ID)/metrics.parquet') GROUP BY ALL ORDER BY concurrency, mode;"

status:
	ssh -T $(REMOTE) 'tail -n 20 $(REMOTE_DIR)/launcher.log; tail -n 1 $(REMOTE_DIR)/status.txt'

fetch-results:
	mkdir -p results/$(RUN_ID)
	rsync -az --exclude '*.bundle' --exclude '*.incomplete' --exclude tmp/ --exclude __pycache__/ $(REMOTE):$(REMOTE_DIR)/ results/$(RUN_ID)/

.PHONY: fetch-profile-data
fetch-profile-data:
	mkdir -p results/$(RUN_ID)
	rsync -az --exclude '*.bundle' --exclude '*.incomplete' --exclude '*.sqlite' --exclude '*.nsys-rep' --exclude profile_export_raw.jsonl --exclude raw_records/ --exclude tmp/ --exclude __pycache__/ $(REMOTE):$(REMOTE_DIR)/ results/$(RUN_ID)/

.PHONY: fetch-model-metadata
fetch-model-metadata:
	mkdir -p results/training-audit/model-metadata
	rsync -azL --include config.json --include train_config.py --include README.md --exclude '*' $(REMOTE):/data/tomasruizt/hub_cache/models--satgeze--Qwen3.6-27B-DSpark/snapshots/47d8de04dd814f4a727bf3fc6a85613d2a3dcaa1/ results/training-audit/model-metadata/
