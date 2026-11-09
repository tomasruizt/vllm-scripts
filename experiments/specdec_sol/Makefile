SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ore-b300-01
REMOTE_ROOT ?= runs
CONFIG ?= $(error set CONFIG, a workload config, see the README)
RUN_ID ?= $(error set RUN_ID, one per attempt, e.g. RUN_ID=qwen36-dspark-20261103)
ESTIMATES ?=  # set to 1 to also price steps and kernels with the analytic and semantic models (default: NCU measurements only)
NCU_FROM ?=  # optional results/<RUN_ID> whose ncu CSVs to reuse for a run without the ncu pass
LOCAL_PYTHON ?= $(HOME)/.venv/bin/python
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10
RUN_DIR = $(REMOTE_ROOT)/$(RUN_ID)
GREEDY_INPUT ?= $(error set GREEDY_INPUT, a glob matching measurements.jsonl for one pinned comparison)
GREEDY_OUTPUT ?= results/greedy-rejection

.PHONY: plot-pareto launch status watch fill-ncu watch-fill step-ncu watch-step fetch-results fetch-greedy-results analyze-greedy analyze report
# Copies the code and config, then runs every concurrency level in parallel, each in its own chg reservation.
launch:
	$(SSH) $(REMOTE) 'mkdir "$(RUN_DIR)"'  # fails if the run exists: pick a new RUN_ID, never delete old runs
	scp -q run.py sol_probe.py ../../benchmarking/host_load.sh $(REMOTE):"$(RUN_DIR)/"
	scp -q $(CONFIG) $(REMOTE):"$(RUN_DIR)/config.yaml"
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)" && setsid nohup ~/.venv/bin/python run.py config.yaml > launch.log 2>&1 < /dev/null & echo launched'

status:
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)"; for f in c*/status.txt; do echo "$$f: $$(tail -n 1 $$f)"; done; tail -n 3 launch.log c*-job.log'

watch:
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)"; timeout 14400 bash -c "until test -f launch-exit-code.txt; do sleep 30; done"; cat launch-exit-code.txt'

# The live ncu pass may not launch every kernel of the nsys run. This replays the nsys run's step shapes under ncu into
# c<N>/ncu_fill/. Needs `make analyze` first (for steps.parquet), then `make fetch-results analyze` to use the new counters.
fill-ncu:
	for d in results/$(RUN_ID)/c*/; do c=$$(basename $$d); \
	  $(LOCAL_PYTHON) fill_shapes.py report/$(RUN_ID)/$$c/steps.parquet results/$(RUN_ID)/$$c/ncu-fill-shapes.json; done
	$(SSH) $(REMOTE) 'mkdir -p "$(RUN_DIR)/fill"'
	scp -q run.py sol_probe.py $(REMOTE):"$(RUN_DIR)/fill/"
	rsync -aq --include='c*/' --include='ncu-fill-shapes.json' --exclude='*' results/$(RUN_ID)/ $(REMOTE):$(RUN_DIR)/
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)/fill" && rm -f ../ncu_fill-exit-code.txt && setsid nohup ~/.venv/bin/python run.py ../config.yaml --only ncu_fill > ../fill.log 2>&1 < /dev/null & echo launched'  # cwd first on the server's sys.path, so not the old sol_probe.py of the run directory

# Counters per launch for one step of every shape, matched by position in the step (c<N>/ncu_step/). Needs `make analyze` first.
step-ncu:
	for d in results/$(RUN_ID)/c*/; do c=$$(basename $$d); \
	  $(LOCAL_PYTHON) fill_shapes.py report/$(RUN_ID)/$$c/steps.parquet results/$(RUN_ID)/$$c/ncu-fill-shapes.json; done
	$(SSH) $(REMOTE) 'mkdir -p "$(RUN_DIR)/fill"'
	scp -q run.py sol_probe.py $(REMOTE):"$(RUN_DIR)/fill/"
	rsync -aq --include='c*/' --include='ncu-fill-shapes.json' --exclude='*' results/$(RUN_ID)/ $(REMOTE):$(RUN_DIR)/
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)/fill" && rm -f ../ncu_step-exit-code.txt && setsid nohup ~/.venv/bin/python run.py ../config.yaml --only ncu_step > ../step.log 2>&1 < /dev/null & echo launched'

watch-step:
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)"; timeout 14400 bash -c "until test -f ncu_step-exit-code.txt; do sleep 30; done"; cat ncu_step-exit-code.txt'

watch-fill:
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)"; timeout 14400 bash -c "until test -f ncu_fill-exit-code.txt; do sleep 30; done"; cat ncu_fill-exit-code.txt'

# The .nsys-rep and .ncu-rep stay remote, the analysis needs only the sqlite export and the ncu CSV.
fetch-results:
	mkdir -p results/$(RUN_ID)
	rsync -az --exclude tmp/ --exclude '*.nsys-rep' --exclude '*.ncu-rep' --exclude '*.qdstrm' --exclude profile_export_raw.jsonl $(REMOTE):$(RUN_DIR)/ results/$(RUN_ID)/

fetch-greedy-results:
	mkdir -p results/$(RUN_ID)
	rsync -az --include='*.jsonl' --include='*.json' --include='*.csv' --include='*.log' --include='*exit-code.txt' --exclude='*' $(REMOTE):$(RUN_DIR)/ results/$(RUN_ID)/

analyze-greedy:
	mkdir -p $(GREEDY_OUTPUT)
	duckdb -cmd "SET VARIABLE input_glob='$(GREEDY_INPUT)'; SET VARIABLE output_dir='$(GREEDY_OUTPUT)';" < analyze_greedy_rejection.sql

analyze:
	for d in results/$(RUN_ID)/c*/; do c=$$(basename $$d); \
	  $(LOCAL_PYTHON) sol.py $$d --output report/$(RUN_ID)/$$c $(if $(ESTIMATES),--estimates) $(if $(NCU_FROM),--ncu $(NCU_FROM)/$$c/ncu/kernels.csv) > /dev/null; done

# Offline HTML with one tab per concurrency level: report/$(RUN_ID)/index.html
report: fetch-results analyze
	$(LOCAL_PYTHON) report.py results/$(RUN_ID) report/$(RUN_ID)

# Throughput-interactivity curves of the serve-pass sweep runs (one server per arm), via the shared AIPerf plotter.
SWEEP_GLOB ?= results/gemma4-sweep-*-2026110*
SWEEP_OUT ?= results/gemma4-sweep-analysis
plot-pareto:
	$(LOCAL_PYTHON) serve_metrics.py '$(SWEEP_GLOB)' $(SWEEP_OUT)
	duckdb -c "COPY (SELECT arm AS mode, concurrency, throughput, 1000 / mean_tpot_ms AS interactivity FROM read_parquet('$(SWEEP_OUT)/metrics.parquet')) TO '$(SWEEP_OUT)/plot-runs.csv' (HEADER);"
	$(LOCAL_PYTHON) ../qwen_dflash_av_60229/build_report.py --plot-csv $(SWEEP_OUT)/plot-runs.csv --output plots/gemma4-sweep --figsize 9 6.4 \
	  --title 'Gemma4-31B-it + DFlash2 · 1× H200 · HumanEval' --series 'baseline:Target only:#898781:s' \
	  --series 'fixed-k3:Fixed K=3:#86b6ef:o' --series 'fixed-k5:Fixed K=5:#2a78d6:o' --series 'fixed-k7:Fixed K=7:#104281:o' \
	  --series 'av-k7:AV K=7, limit 512 (default):#eb6834:D' --series 'av-k7-cap1536:AV K=7, limit 1024:#1baf7a:D'
