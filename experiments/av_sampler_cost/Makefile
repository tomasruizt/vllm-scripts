SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ohio-b200-01
RUN_ID ?= $(error set RUN_ID, one per attempt, e.g. RUN_ID=av-sampler-4b-20261125-1)
GPU ?= $(error set GPU, a free GPU id on $(REMOTE))
# e.g. EXTRA_ENV='MODEL=Qwen/Qwen3.8-27B SPEC_MODEL=RedHatAI/Qwen3.8-27B-speculator.dflash2 K=7 CONCURRENCIES=16,64'
EXTRA_ENV ?=
TIMEOUT ?= 90m
RUN = runs/$(RUN_ID)
SSH := ssh -T -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30

.PHONY: launch status watch fetch-results analyze
launch:
	$(SSH) $(REMOTE) 'mkdir -p runs && mkdir "$(RUN)"'  # fails if the run exists: pick a new RUN_ID
	scp -q run.sh ../av_alloc_headroom/make_prompts.py $(REMOTE):"$(RUN)/"
	$(SSH) $(REMOTE) 'cd "$(RUN)" && HF_HOME=/data/tomasruizt/huggingface ~/.venv/bin/python make_prompts.py prompts mixed'
	$(SSH) $(REMOTE) 'cd "$(RUN)" && setsid nohup chg run --gpu-ids $(GPU) --timeout $(TIMEOUT) --note "AV sampler cost" -- env $(EXTRA_ENV) bash run.sh $(GPU) > job.log 2>&1 < /dev/null & echo launched'

status:
	$(SSH) $(REMOTE) 'cd "$(RUN)"; grep CELL job.log | tail -n 1; grep -c . steps.jsonl.* | tail -n 3; cat failures.txt 2>/dev/null || true'

watch:
	$(SSH) $(REMOTE) 'cd "$(RUN)"; timeout 7000 bash -c "until test -f exit-code.txt; do sleep 20; done"; cat exit-code.txt; tail -n 3 job.log'

fetch-results:
	mkdir -p results/$(RUN_ID)
	rsync -az --exclude prompts --exclude inputs.json --exclude profile_export_raw.jsonl --exclude gpu_telemetry_export.jsonl --exclude '*.py' $(REMOTE):$(RUN)/ results/$(RUN_ID)/

# Builds results/steps.parquet (all fetched runs, long format) and prints the tables.
analyze:
	~/.venv/bin/python analyze.py results
	duckdb < report.sql | tee results/report.txt

# One server, a short torch-profiler trace per sampling mode. Then: make fetch-results, prof_sample.py results/$(RUN_ID).
.PHONY: launch-profile
launch-profile:
	$(SSH) $(REMOTE) 'mkdir -p runs && mkdir "$(RUN)"'
	scp -q profile.sh ../av_alloc_headroom/make_prompts.py $(REMOTE):"$(RUN)/"
	$(SSH) $(REMOTE) 'cd "$(RUN)" && HF_HOME=/data/tomasruizt/huggingface ~/.venv/bin/python make_prompts.py prompts mixed'
	$(SSH) $(REMOTE) 'cd "$(RUN)" && setsid nohup chg run --gpu-ids $(GPU) --timeout $(TIMEOUT) --note "AV sampler cost profile" -- env $(EXTRA_ENV) bash profile.sh $(GPU) > job.log 2>&1 < /dev/null & echo launched'
