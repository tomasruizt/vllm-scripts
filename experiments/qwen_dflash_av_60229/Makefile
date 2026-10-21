SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := all

REMOTE ?= RH-nm-ore-b300-01
REMOTE_RUNS ?= /home/tomasruizt/runs
RUN_ID ?= qwen-speedbench-steady-20261016
REMOTE_RUN_DIR := $(REMOTE_RUNS)/$(RUN_ID)
REMOTE_EXP_DIR := $(REMOTE_RUNS)/qwen-dflash-av-60229
GPUS ?= 1 2
REPEATS ?= 3
PILOT_CELLS ?= 1:24 64:320 128:512
REMOTE_REPORT_DIR := $(REMOTE_RUNS)/$(RUN_ID)-report
REMOTE_BUILDER_DIR ?= $(REMOTE_RUNS)/qwen-av-report-builder
REMOTE_PYTHON ?= $$HOME/.venv/bin/python
DUCKDB ?= duckdb

RESULTS_DIR := results/$(RUN_ID)
REPORT_DIR ?= report
REPORT_STAGE := /private/tmp/$(RUN_ID)-report-stage
QUICK_PARQUET := /private/tmp/$(RUN_ID)-quick.parquet
MERGED_PARQUET := /private/tmp/$(RUN_ID)-metrics.parquet

.PHONY: all report sync fetch-results parquet help launch pilot status

.PHONY: fetch-cost-results
.PHONY: fetch-community-gdn analyze-community-gdn
.PHONY: fetch-quick-profiling analyze-quick-profiling
.PHONY: fetch-profiling-sweep analyze-profiling-sweep plot-profiling-sweep
plot-profiling-sweep:
	$(DUCKDB) -c "COPY (SELECT arm AS mode, concurrency, throughput, 1000.0 / mean_tpot_ms AS interactivity FROM read_parquet('results/qwen-gdn-repeated-serving-20261020/metrics.parquet') WHERE valid ORDER BY arm, concurrency, worker, trial) TO 'results/qwen-gdn-repeated-serving-20261020/plot-runs.csv' (HEADER);"
	$(LOCAL_PYTHON) build_report.py --plot-csv results/qwen-gdn-repeated-serving-20261020/plot-runs.csv --output plots/repeated-profiling --figsize 7.6 6.4 --title 'Qwen3.5-35B-A3B + DFlash' --series 'main:GDN + AV:#64748b:o' --series 'community:GDN + AV + #54065:#c44e45:s' --series 'gdn:GDN + AV + #54065 + #62715:#138b83:D'

fetch-profiling-sweep:
	mkdir -p results/qwen-gdn-repeated-serving-20261020
	rsync -az --exclude inputs.json --exclude profile_export_raw.jsonl --exclude gpu_telemetry_export.jsonl --exclude output_fragments --exclude raw_records $(REMOTE):$(REMOTE_RUNS)/qwen-gdn-repeated-serving-20261020/ results/qwen-gdn-repeated-serving-20261020/

analyze-profiling-sweep:
	$(LOCAL_PYTHON) summarize_profiling_sweep.py results/qwen-gdn-repeated-serving-20261020 $(SWEEP_CHECK)
	$(DUCKDB) -c "SELECT concurrency, arm, count(*) AS trials, count(*) FILTER (WHERE valid) AS valid_trials, round(median(throughput) FILTER (WHERE valid)) AS median_tok_s, round(min(throughput) FILTER (WHERE valid)) AS min_tok_s, round(max(throughput) FILTER (WHERE valid)) AS max_tok_s, round(median(mean_tpot_ms) FILTER (WHERE valid), 2) AS median_tpot_ms, round(median(mean_itl_ms) FILTER (WHERE valid), 2) AS median_itl_ms, round(median(mean_ttft_ms) FILTER (WHERE valid), 1) AS median_ttft_ms FROM read_parquet('results/qwen-gdn-repeated-serving-20261020/metrics.parquet') GROUP BY ALL ORDER BY concurrency, arm;"

fetch-quick-profiling:
	$(MAKE) fetch-results RUN_ID=qwen-gdn-quick-serving-20261020

analyze-quick-profiling:
	$(LOCAL_PYTHON) summarize_quick_profiling.py results/qwen-gdn-quick-serving-20261020
	$(DUCKDB) -c "SELECT arm, concurrency, round(throughput) AS tok_s, round(mean_tpot_ms, 2) AS tpot_ms, round(mean_itl_ms, 2) AS itl_ms, round(mean_ttft_ms, 1) AS ttft_ms, round(acceptance_length, 2) AS acceptance_length, round(window_s, 1) AS steady_s, preemptions, valid, failed_gates FROM read_parquet('results/qwen-gdn-quick-serving-20261020/metrics.parquet') ORDER BY concurrency, arm;"

fetch-community-gdn:
	$(MAKE) fetch-results RUN_ID=qwen-gdn-community-20261019

analyze-community-gdn:
	MPLCONFIGDIR=/private/tmp/rh-matplotlib $(LOCAL_PYTHON) plot_serving_cost_preview.py
	$(DUCKDB) < community_gdn.sql

.PHONY: fetch-tail-profiles analyze-tail-profiles
fetch-tail-profiles:
	$(MAKE) fetch-results RUN_ID=qwen-av-tail-profile-20261019

analyze-tail-profiles:
	$(LOCAL_PYTHON) analyze_tail_profiles.py
	$(DUCKDB) < tail_profiles.sql
	$(LOCAL_PYTHON) plot_tail_profiles.py

.PHONY: fetch-raw-cost-curves analyze-raw-cost-curves
fetch-raw-cost-curves:
	$(MAKE) fetch-results RUN_ID=qwen-gdn-profile-fix-20261019

analyze-raw-cost-curves:
	$(DUCKDB) < raw_cost_curves.sql
	$(LOCAL_PYTHON) plot_raw_cost_curves.py

.PHONY: fetch-query-results analyze-query-shapes
fetch-query-results:
	$(MAKE) fetch-results RUN_ID=qwen-q1q2-20261019

analyze-query-shapes:
	$(LOCAL_PYTHON) analyze_query_shapes.py results/qwen-q1q2-20261019
	$(DUCKDB) -c "COPY (SELECT * FROM read_json_auto('results/qwen-q1q2-20261019/timings.json')) TO 'results/qwen-q1q2-20261019/metrics.parquet' (FORMAT PARQUET); COPY (SELECT * FROM read_json_auto('results/qwen-q1q2-20261019/kernels.json')) TO 'results/qwen-q1q2-20261019/kernels.parquet' (FORMAT PARQUET);"

fetch-cost-results:
	$(MAKE) fetch-results RUN_ID=qwen-cost-calibration-20261019
	$(MAKE) fetch-results RUN_ID=qwen-cost-fixed-20261019
	$(MAKE) fetch-results RUN_ID=qwen-cost-corrected-20261019
	$(MAKE) fetch-results RUN_ID=qwen-cost-startup-20261019
	$(MAKE) fetch-results RUN_ID=qwen-cost-matched-20261019

COST_OUTPUT ?= results/cost-curve-analysis-20261019
FIXED_DRAFTER_OUTPUT ?= results/fixed-drafter-analysis-20261019
ORACLE_OUTPUT ?= results/cost-oracle-analysis-20261019
ORACLE_POLICY_OUTPUT ?= results/cost-oracle-policy-analysis-20261019
LOCAL_PYTHON ?= $(HOME)/.venv/bin/python
C256_OUTPUT ?= results/cost-c256-analysis-20261019
C256_PREFIX ?= qwen-cost-c256
C256_TP ?= 1
C256_TP2_OUTPUT ?= results/cost-c256-tp2-analysis-20261019
FULL_CURVE_RUN ?= qwen-full-curve-c256-20261020
AV_FULL_SWEEP_RUN ?= qwen-av-full-sweep-20261020

.PHONY: fetch-av-full-sweep-results analyze-av-full-sweep analyze-av-full-sweep-acceptance plot-av-full-sweep
fetch-av-full-sweep-results:
	$(MAKE) fetch-results RUN_ID=$(AV_FULL_SWEEP_RUN)

analyze-av-full-sweep:
	$(LOCAL_PYTHON) summarize_av_full_sweep.py

analyze-av-full-sweep-acceptance:
	$(DUCKDB) < acceptance_full_sweep.sql

plot-av-full-sweep:
	$(LOCAL_PYTHON) plot_av_full_sweep.py

.PHONY: fetch-full-curve-results analyze-full-curve
fetch-full-curve-results:
	$(MAKE) fetch-results RUN_ID=$(FULL_CURVE_RUN)

analyze-full-curve:
	$(LOCAL_PYTHON) summarize_full_curve.py

.PHONY: fetch-c256-results analyze-c256
.PHONY: fetch-c256-tp2-results fetch-c256-frozen analyze-c256-tp2
fetch-c256-tp2-results:
	$(MAKE) fetch-c256-results C256_PREFIX=qwen-cost-c256-tp2

analyze-c256-tp2:
	$(MAKE) analyze-c256 C256_PREFIX=qwen-cost-c256-tp2 C256_TP=2 C256_OUTPUT=$(C256_TP2_OUTPUT)

fetch-c256-frozen:
	mkdir -p $(C256_TP2_OUTPUT)
	rsync -az $(REMOTE):$(REMOTE_RUNS)/qwen-cost-oracle-tools-20261019/c256-frozen/cost-lookup-frozen.json $(C256_TP2_OUTPUT)/
	rsync -az $(REMOTE):$(REMOTE_RUNS)/qwen-cost-oracle-tools-20261019/c256-frozen/points-frozen.parquet $(C256_TP2_OUTPUT)/

fetch-c256-results:
	mkdir -p results/$(C256_PREFIX)-{a,b}-20261019
	rsync -az --delete --exclude inputs.json --exclude profile_export_raw.jsonl --exclude gpu_telemetry_export.jsonl --exclude output_fragments --exclude raw_records $(REMOTE):$(REMOTE_RUNS)/$(C256_PREFIX)-a-20261019/ results/$(C256_PREFIX)-a-20261019/
	rsync -az --delete --exclude inputs.json --exclude profile_export_raw.jsonl --exclude gpu_telemetry_export.jsonl --exclude output_fragments --exclude raw_records $(REMOTE):$(REMOTE_RUNS)/$(C256_PREFIX)-b-20261019/ results/$(C256_PREFIX)-b-20261019/

analyze-c256:
	$(LOCAL_PYTHON) summarize_cost_c256.py --output $(C256_OUTPUT) --prefix $(C256_PREFIX) --tp $(C256_TP)
COST_ARGS := --baseline results/qwen-speedbench-steady-cg-20261019
COST_ARGS += --fixed results/qwen-cost-fixed-20261019
COST_ARGS += --corrected results/qwen-cost-corrected-20261019
COST_ARGS += --calibration results/qwen-cost-calibration-20261019
COST_ARGS += --startup results/qwen-cost-startup-20261019
COST_ARGS += --matched results/qwen-cost-matched-20261019

.PHONY: fit-cost-curves summarize-cost
.PHONY: fetch-cost-oracle-results analyze-cost-oracle
fetch-cost-oracle-results:
	$(MAKE) fetch-results RUN_ID=qwen-cost-oracle-a-20261019
	$(MAKE) fetch-results RUN_ID=qwen-cost-oracle-b-20261019

analyze-cost-oracle:
	mkdir -p $(ORACLE_OUTPUT)
	sed 's|__OUTPUT__|$(ORACLE_OUTPUT)|g' cost_oracle.sql | $(DUCKDB)
	$(LOCAL_PYTHON) summarize_cost_oracle.py --output $(ORACLE_OUTPUT)

.PHONY: fetch-cost-oracle-policy-results summarize-cost-oracle-policy
fetch-cost-oracle-policy-results:
	$(MAKE) fetch-results RUN_ID=qwen-av-cost-oracle-policy-a-20261019
	$(MAKE) fetch-results RUN_ID=qwen-av-cost-oracle-policy-b-20261019

summarize-cost-oracle-policy:
	$(LOCAL_PYTHON) summarize_cost_oracle_policy.py --output $(ORACLE_POLICY_OUTPUT)

.PHONY: fetch-fixed-drafter-results fit-fixed-drafter summarize-fixed-drafter
.PHONY: compare-profile-runtime
compare-profile-runtime:
	$(DUCKDB) < compare_profile_runtime.sql

fetch-fixed-drafter-results:
	$(MAKE) fetch-results RUN_ID=qwen-fixed-drafter-calibration-20261019
	$(MAKE) fetch-results RUN_ID=qwen-fixed-drafter-controls-20261019

fit-fixed-drafter:
	$(LOCAL_PYTHON) fit_cost_curves.py results/qwen-fixed-drafter-calibration-20261019 --output $(FIXED_DRAFTER_OUTPUT) --holdout-drafts 3

summarize-fixed-drafter:
	$(LOCAL_PYTHON) summarize_fixed_drafter.py --calibration results/qwen-fixed-drafter-calibration-20261019 --controls results/qwen-fixed-drafter-controls-20261019 --output $(FIXED_DRAFTER_OUTPUT)
	sed 's|__OUTPUT__|$(FIXED_DRAFTER_OUTPUT)|g' cost_analysis.sql | $(DUCKDB)
	sed 's|__OUTPUT__|$(FIXED_DRAFTER_OUTPUT)|g' fixed_drafter_analysis.sql | $(DUCKDB)

fit-cost-curves:
	$(LOCAL_PYTHON) fit_cost_curves.py results/qwen-cost-calibration-20261019 --output $(COST_OUTPUT)

summarize-cost:
	$(LOCAL_PYTHON) summarize_cost_experiment.py $(COST_ARGS) --output $(COST_OUTPUT)
	sed 's|__OUTPUT__|$(COST_OUTPUT)|g' cost_analysis.sql | $(DUCKDB)

all: parquet

launch:
	scp -q run-steady.sh steady-speedbench.yaml $(REMOTE):$(REMOTE_EXP_DIR)/
	ssh -T -o BatchMode=yes $(REMOTE) 'mkdir -p "$(REMOTE_RUN_DIR)" && ROOT="$(REMOTE_RUN_DIR)" $(if $(CELLS),CELLS="$(CELLS)") $(if $(PLAN_A),PLAN_A="$(PLAN_A)" PLAN_B="$(PLAN_B)") nohup bash "$(REMOTE_EXP_DIR)/run-steady.sh" $(GPUS) $(REPEATS) > "$(REMOTE_RUN_DIR)/launch.log" 2>&1 < /dev/null &'
	echo "Launched $(RUN_ID) on GPUs $(GPUS). Check with: make status RUN_ID=$(RUN_ID)"

pilot:
	$(MAKE) launch RUN_ID=$(RUN_ID)-pilot REPEATS=1 CELLS="$(PILOT_CELLS)"

status:
	ssh -T -o BatchMode=yes $(REMOTE) 'tail -n 3 "$(REMOTE_RUN_DIR)"/worker-gpu*.log; ls "$(REMOTE_RUN_DIR)"/worker-gpu*-exit-code.txt 2>/dev/null || true'

help:
	echo 'make all       sync remote AIPerf data, update metrics.parquet, and rebuild report/'
	echo 'make report    rebuild report/ from the remote AIPerf artifacts'
	echo 'make fetch-results  copy remote AIPerf results to results/ (no report rebuild)'
	echo 'make parquet  sync data and merge this suite into metrics.parquet with DuckDB'
	echo 'make fetch-cost-results  fetch the runtime-calibration experiment and controls'
	echo 'make fit-cost-curves     fit measured decode costs and plot against startup costs'
	echo 'make summarize-cost     collect all cost-curve arms into Parquet and CSV'
	echo 'Override REMOTE, RUN_ID, REMOTE_RUNS, or DUCKDB as needed.'

report:
	ssh -T -o BatchMode=yes $(REMOTE) 'mkdir -p "$(REMOTE_BUILDER_DIR)/reporting" "$(REMOTE_BUILDER_DIR)/experiments/qwen_dflash_av_60229"'
	scp -q build_report.py $(REMOTE):$(REMOTE_BUILDER_DIR)/experiments/qwen_dflash_av_60229/build_report.py
	scp -q ../../reporting/plots.py $(REMOTE):$(REMOTE_BUILDER_DIR)/reporting/plots.py
	scp -q ../../reporting/b200-theme.css $(REMOTE):$(REMOTE_BUILDER_DIR)/reporting/b200-theme.css
	ssh -T -o BatchMode=yes $(REMOTE) 'rm -rf "$(REMOTE_REPORT_DIR)" && "$(REMOTE_PYTHON)" "$(REMOTE_BUILDER_DIR)/experiments/qwen_dflash_av_60229/build_report.py" --source "$(REMOTE_RUN_DIR)" --output "$(REMOTE_REPORT_DIR)" $(REPORT_ARGS)'
	rm -rf "$(REPORT_STAGE)"
	mkdir -p "$(REPORT_STAGE)"
	ssh -T -o BatchMode=yes $(REMOTE) 'tar czf - -C "$(REMOTE_RUNS)" "$(notdir $(REMOTE_REPORT_DIR))"' | tar -xzf - -C "$(REPORT_STAGE)"
	rm -rf "$(REPORT_DIR)"
	mv "$(REPORT_STAGE)/$(notdir $(REMOTE_REPORT_DIR))" "$(REPORT_DIR)"
	echo "Updated $(REPORT_DIR)/index.html"

sync: report fetch-results

fetch-results:
	mkdir -p results
	ssh -T -o BatchMode=yes $(REMOTE) 'tar czf - --exclude="inputs.json" --exclude="profile_export_raw.jsonl" --exclude="gpu_telemetry_export.jsonl" --exclude="*.csv" -C "$(REMOTE_RUNS)" "$(RUN_ID)"' | tar -xzf - -C results
	echo "Synced $(RESULTS_DIR)"

parquet: sync
	$(DUCKDB) -c "COPY (SELECT *, throughput AS output_throughput, duration_s AS duration, 1000.0 / mean_tpot_ms AS interactivity_tok_s_user, CASE WHEN error_rate = 0 THEN requests ELSE NULL END AS completed FROM read_json_auto('$(RESULTS_DIR)/quick-result.json', format='array')) TO '$(QUICK_PARQUET)' (FORMAT PARQUET, COMPRESSION ZSTD);"
	$(DUCKDB) -c "COPY (SELECT * FROM read_parquet('metrics.parquet') WHERE suite IS DISTINCT FROM '$(RUN_ID)' UNION ALL BY NAME SELECT * FROM read_parquet('$(QUICK_PARQUET)')) TO '$(MERGED_PARQUET)' (FORMAT PARQUET, COMPRESSION ZSTD);"
	mv "$(MERGED_PARQUET)" metrics.parquet
	rm -f "$(QUICK_PARQUET)"
	$(DUCKDB) -csv metrics.parquet -c "SELECT count(*) AS total_rows, count(*) FILTER (WHERE suite = '$(RUN_ID)') AS suite_rows FROM read_parquet('metrics.parquet');"

PROF_RUN_DIR := $(REMOTE_RUNS)/qwen-prefill-step-profile
PROF_MODE ?= torch

.PHONY: profile-prefill fetch-profile
profile-prefill:
	ssh -T -o BatchMode=yes $(REMOTE) 'mkdir -p "$(PROF_RUN_DIR)"'
	scp -q profile-prefill-step.sh $(REMOTE):$(PROF_RUN_DIR)/
	ssh -T -o BatchMode=yes $(REMOTE) 'cd "$(PROF_RUN_DIR)" && ROOT="$(PROF_RUN_DIR)" nohup canhazgpu run --gpus 1 --timeout 1h --note "prefill step profile" -- bash profile-prefill-step.sh $(PROF_MODE) > run.log 2>&1 < /dev/null &'

fetch-profile:
	mkdir -p results/prefill-step-profile
	rsync -a $(REMOTE):$(PROF_RUN_DIR)/{traces,server-command.txt,run.log,pyspy-prefill.txt} results/prefill-step-profile/

NATIVE_TABLES_RUN ?= qwen-native-tables-tp2-20261021

.PHONY: fetch-native-tables plot-cost-curves
fetch-native-tables:
	$(MAKE) fetch-results RUN_ID=$(NATIVE_TABLES_RUN)

plot-cost-curves:
	$(LOCAL_PYTHON) plot_base_vs_pr_cost_curves.py --draft --output plots/full-curve-sweep/cost-curves \
		--arm "results/$(NATIVE_TABLES_RUN)/cost-tables.txt:Native (startup profile):#d97a00" \
		--arm "results/$(FULL_CURVE_RUN)/measured-curves-frozen.json:Measured (serving):#13706a" \
		--verify-label "Target forward cost (ms)" \
		--title "AV cost tables, native startup profile vs measured during serving (Qwen3.5-35B-A3B + DFlash k=7, 2x B300 TP2)"
