SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ore-b300-01
RUN_ID ?= topk-topp-20261103
BENCH_ARGS ?=
SCRIPT ?= benchmark.py
CAPTURE_SCRIPT ?= capture.py
BENCHMARK_SOURCE ?=
EXTRA_SOURCES ?=
VLLM_ROOT ?= $(HOME)/code/rh/vllm
REMOTE_PYTHON ?= ~/code/vllm/.venv/bin/python
GPU_ARGS ?= --gpus 1
SWEEP_TIMEOUT ?= 45m
REMOTE_DIR := runs/$(RUN_ID)
SSH_ARGS := -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none
SSH := ssh -T $(SSH_ARGS)
SCP := scp -q $(SSH_ARGS)

.PHONY: launch launch-existing launch-sweep launch-capture fetch-results fetch-expanded analyze analyze-existing analyze-expanded analyze-reduced analyze-compact analyze-validation analyze-serving status
launch:
	$(SSH) $(REMOTE) 'mkdir "$(REMOTE_DIR)"'
	$(SCP) *.py $(REMOTE):"$(REMOTE_DIR)/"
	if [[ -n "$(BENCHMARK_SOURCE)" ]]; then $(SCP) "$(BENCHMARK_SOURCE)" $(REMOTE):"$(REMOTE_DIR)/benchmark_topk_topp.py"; fi
	if [[ -n "$(EXTRA_SOURCES)" ]]; then $(SCP) $(EXTRA_SOURCES) $(REMOTE):"$(REMOTE_DIR)/"; fi
	$(SSH) $(REMOTE) 'cd "$(REMOTE_DIR)" && setsid nohup bash -c '\''remote_python=$(REMOTE_PYTHON); export PATH="$$(dirname "$$remote_python"):$$PATH"; chg run $(GPU_ARGS) --timeout 30m --note topk-topp-microbench -- "$$remote_python" $(SCRIPT) --output results $(BENCH_ARGS); code=$$?; echo $$code > exit-code.txt'\'' > launch.log 2>&1 < /dev/null &'

launch-existing:
	$(SSH) $(REMOTE) 'mkdir "$(REMOTE_DIR)"'
	$(SCP) $(VLLM_ROOT)/benchmarks/benchmark_topk_topp.py run_existing.sh $(REMOTE):"$(REMOTE_DIR)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_DIR)" && setsid nohup bash -c '\''chg run --gpus 1 --timeout 30m --note vllm-topk-topp-benchmark -- bash run_existing.sh $(BENCH_ARGS); code=$$?; echo $$code > exit-code.txt'\'' > launch.log 2>&1 < /dev/null &'

launch-sweep:
	$(SSH) $(REMOTE) 'mkdir -p runs && mkdir "$(REMOTE_DIR)"'
	$(SCP) $(VLLM_ROOT)/benchmarks/benchmark_topk_topp.py sweep_existing.py $(REMOTE):"$(REMOTE_DIR)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_DIR)" && setsid nohup bash -c '\''remote_python=$(REMOTE_PYTHON); export PATH="$$(dirname "$$remote_python"):$$PATH"; chg run $(GPU_ARGS) --timeout $(SWEEP_TIMEOUT) --note vllm-topk-topp-expanded -- "$$remote_python" sweep_existing.py --output results $(BENCH_ARGS); code=$$?; echo $$code > exit-code.txt'\'' > launch.log 2>&1 < /dev/null &'

launch-capture:
	$(SSH) $(REMOTE) 'mkdir "$(REMOTE_DIR)"'
	$(SCP) capture.py serving.py benchmark.py ../specdec_sol/run.py $(REMOTE):"$(REMOTE_DIR)/"
	$(SCP) capture_probe.py $(REMOTE):"$(REMOTE_DIR)/sol_probe.py"
	$(SCP) ../specdec_sol/results/qwen36-dspark-av-main-sampling-20261103/config.yaml $(REMOTE):"$(REMOTE_DIR)/config.yaml"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_DIR)" && setsid nohup bash -c '\''export PATH="$$HOME/code/vllm/.venv/bin:$$PATH"; chg run --gpus 1 --timeout 30m --note topk-topp-real-logits -- ~/code/vllm/.venv/bin/python $(CAPTURE_SCRIPT) --output results; code=$$?; echo $$code > exit-code.txt'\'' > launch.log 2>&1 < /dev/null &'

# Kept-token prototype: tests + benchmark of the local worktree's diff against its base commit.
KEPT_WORKTREE ?= $(HOME)/code/rh/vllm-topk-sparse
KEPT_BASE ?= ~/code/vllm-pr-58340
launch-kept:
	$(SSH) $(REMOTE) 'mkdir -p "$(REMOTE_DIR)"'
	git -C $(KEPT_WORKTREE) diff HEAD | $(SSH) $(REMOTE) 'cat > "$(REMOTE_DIR)/kept-tokens.patch"'
	git -C $(KEPT_WORKTREE) rev-parse HEAD | $(SSH) $(REMOTE) 'cat > "$(REMOTE_DIR)/commit.txt"'
	$(SCP) kept_tokens.sh kept_tokens_bench.py $(REMOTE):$(REMOTE_DIR)/
	$(SSH) -n $(REMOTE) 'cd "$(REMOTE_DIR)" && setsid nohup bash -c '\''bash kept_tokens.sh $(KEPT_BASE) $$(cat commit.txt) $$PWD/kept-tokens.patch $$PWD/results; echo $$? > exit-code.txt'\'' > launch.log 2>&1 < /dev/null &'

fetch-results:
	mkdir -p results/$(RUN_ID)
	rsync -az -e "$(SSH)" --exclude tmp/ $(REMOTE):$(REMOTE_DIR)/ results/$(RUN_ID)/

fetch-expanded:
	for shard in '' -s0 -s1 -s2; do $(MAKE) fetch-results RUN_ID=topk-topp-expanded-b300$$shard-20261104; done
	$(MAKE) fetch-results REMOTE=RH-nm-automation-h100-standalone-1-preserve RUN_ID=topk-topp-expanded-h100-frk-20261104
	$(MAKE) fetch-results REMOTE=RH-nm-automation-h100-standalone-1-preserve RUN_ID=topk-topp-expanded-h100-frk-r2-20261104

analyze-expanded:
	mkdir -p results/topk-topp-expanded-analysis-20261104
	duckdb < expanded.sql

analyze-reduced:
	duckdb -cmd "SET VARIABLE reduced_dir='results/$(RUN_ID)';" < reduced.sql

analyze-uniform-k:
	duckdb -cmd "SET VARIABLE ablation_dir='results/$(RUN_ID)';" < uniform_k.sql

analyze-scenarios:
	duckdb -cmd "SET VARIABLE scenario_dir='results/$(RUN_ID)';" < scenarios.sql

launch-renorm:
	$(SSH) $(REMOTE) 'mkdir -p runs && mkdir "$(REMOTE_DIR)"'
	$(SCP) $(BENCHMARK_SOURCE) $(REMOTE):"$(REMOTE_DIR)/benchmark_topk_topp.py"
	$(SCP) renorm_impls.py sweep_renorm.py validate_renorm.py sweep_existing.py run_renorm.sh $(REMOTE):"$(REMOTE_DIR)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_DIR)" && setsid nohup bash -c '\''remote_python=$(REMOTE_PYTHON); export PATH="$$(dirname "$$remote_python"):$$PATH"; chg run $(GPU_ARGS) --timeout $(SWEEP_TIMEOUT) --note topk-topp-renorm -- bash run_renorm.sh "$$remote_python" $(BENCH_ARGS); code=$$?; echo $$code > exit-code.txt'\'' > launch.log 2>&1 < /dev/null &'

launch-renorm-probe:
	$(SSH) $(REMOTE) 'mkdir -p runs && mkdir "$(REMOTE_DIR)"'
	$(SCP) $(BENCHMARK_SOURCE) $(REMOTE):"$(REMOTE_DIR)/benchmark_topk_topp.py"
	$(SCP) renorm_impls.py validate_renorm.py probe_renorm.py $(REMOTE):"$(REMOTE_DIR)/"
	$(SSH) -n $(REMOTE) 'cd "$(REMOTE_DIR)" && setsid nohup bash -c '\''remote_python=$(REMOTE_PYTHON); export PATH="$$(dirname "$$remote_python"):$$PATH"; chg run $(GPU_ARGS) --timeout 10m --note topk-topp-renorm-probe -- "$$remote_python" probe_renorm.py --output results/probe.jsonl --captures $$HOME/runs/topk-topp-real-logits-20261103; code=$$?; echo $$code > exit-code.txt'\'' > launch.log 2>&1 < /dev/null &'

analyze-renorm:
	duckdb -cmd "SET VARIABLE renorm_dir='results/$(RUN_ID)';" < renorm.sql

analyze-optimization:
	duckdb -cmd "SET VARIABLE optimization_dir='results/$(RUN_ID)';" < optimization.sql

analyze-unified:
	duckdb -cmd "SET VARIABLE unified_dir='results/$(RUN_ID)';" < unified.sql

analyze:
	duckdb -c "COPY (SELECT '$(RUN_ID)' AS run_id, * EXCLUDE (kernel_us) FROM read_json_auto('results/$(RUN_ID)/results/metrics.json')) TO 'results/$(RUN_ID)/metrics.parquet' (FORMAT PARQUET); COPY (SELECT '$(RUN_ID)' AS run_id, batch, vocab, distribution, implementation, generate_subscripts(cupti_us, 1) AS repeat, unnest(cupti_us) AS api_span_us FROM read_json_auto('results/$(RUN_ID)/results/metrics.json')) TO 'results/$(RUN_ID)/samples.parquet' (FORMAT PARQUET);"

analyze-existing:
	duckdb -cmd "SET VARIABLE existing_dir='results/$(RUN_ID)';" < existing.sql

analyze-compact: analyze
	duckdb -c "COPY (SELECT run_id, batch, vocab, distribution, implementation, 'eager' AS method, generate_subscripts(eager_us, 1) AS repeat, unnest(eager_us) AS duration_us FROM 'results/$(RUN_ID)/metrics.parquet' UNION ALL SELECT run_id, batch, vocab, distribution, implementation, 'cupti-span' AS method, generate_subscripts(cupti_us, 1) AS repeat, unnest(cupti_us) AS duration_us FROM 'results/$(RUN_ID)/metrics.parquet') TO 'results/$(RUN_ID)/timing-samples.parquet' (FORMAT PARQUET);"

analyze-validation:
	mkdir -p results/topk-topp-validation-20261103
	duckdb < validation.sql

analyze-serving:
	PYTHONPATH=../.. ~/.venv/bin/python analyze_serving.py results/$(RUN_ID)
	duckdb -cmd "SET VARIABLE serving_dir='results/$(RUN_ID)/analysis';" < serving.sql

status:
	$(SSH) $(REMOTE) 'cd "$(REMOTE_DIR)"; tail -n 10 launch.log; tail -n 1 exit-code.txt'

# Hierarchical CUDA filter (hier_dev.sh runs): jsonl into results/hier-cuda/{b300,h100}.
fetch-hier:
	mkdir -p results/hier-cuda/b300 results/hier-cuda/h100
	rsync -az -e "$(SSH)" --include '*.jsonl' --exclude '*' RH-nm-ore-b300-01:runs/hier-dev/ results/hier-cuda/b300/
	rsync -az -e "$(SSH)" --include '*.jsonl' --exclude '*' RH-nm-automation-h100-standalone-1-preserve:runs/hier-dev/ results/hier-cuda/h100/
