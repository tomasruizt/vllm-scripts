SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-ore-b300-01
RUN_ID ?= $(error set RUN_ID, one per attempt, e.g. RUN_ID=topk-topp-e2e-20261106-1)
VARIANTS ?= main-k7 pr-k7
# Sequential runs one variant after the other on a single GPU (e.g. when only one H100 is free).
SEQUENTIAL ?= 0
RUN_DIR = runs/$(RUN_ID)
SSH := ssh -T -n -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none
H100 := RH-nm-automation-h100-standalone-1-preserve
ifeq ($(REMOTE),$(H100))
HOST_ENV := MODEL=RedHatAI/Qwen3.6-35B-A3B-FP8 MAIN=$$HOME/code/topk-cuda-main HUB_CACHE=/mnt/nvme_stripe/playground/hub_cache DATASET=$$HOME/datasets/throughput_1k.jsonl
endif
GPUS ?= 1
CHG_TIMEOUT ?= 240m
CHG := chg run --gpus $(GPUS) --timeout $(CHG_TIMEOUT)

.PHONY: launch status watch fetch-results summarize
# One chg reservation (GPUS GPUs) per variant, all in parallel (or one reservation for all when SEQUENTIAL=1).
launch:
	$(SSH) $(REMOTE) 'mkdir "$(RUN_DIR)"'
	scp -q run.sh aime_eval.py benchmark.yaml benchmark-default.yaml benchmark-glm.yaml ../../benchmarking/host_load.sh "$(REMOTE):$(RUN_DIR)/"
ifeq ($(SEQUENTIAL),1)
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)" && setsid nohup $(CHG) --note "top-k/top-p E2E" -- bash -c "for v in $(VARIANTS); do env $(HOST_ENV) $(EXTRA_ENV) bash run.sh \$$v > \$$v-job.log 2>&1; done" > job.log 2>&1 < /dev/null & echo launched'
else
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)" && for v in $(VARIANTS); do setsid nohup $(CHG) --note "top-k/top-p E2E $$v" -- env $(HOST_ENV) $(EXTRA_ENV) bash run.sh $$v > $$v-job.log 2>&1 < /dev/null & sleep 10; done; echo launched'
endif

status:
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)"; for f in *.log; do echo "$$f: $$(tail -n 1 $$f)"; done'

watch:
	$(SSH) $(REMOTE) 'cd "$(RUN_DIR)"; timeout 14400 bash -c "until test \$$(ls */exit-code.txt 2>/dev/null | wc -l) -ge $(words $(VARIANTS)); do sleep 30; done"; grep . */exit-code.txt'

fetch-results:
	mkdir -p results/$(RUN_ID)
	rsync -az --exclude inputs.json --exclude profile_export_raw.jsonl --exclude gpu_telemetry_export.jsonl $(REMOTE):$(RUN_DIR)/ results/$(RUN_ID)/

summarize:
	~/.venv/bin/python summarize.py results/$(RUN_ID)
