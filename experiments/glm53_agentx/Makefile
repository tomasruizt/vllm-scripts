# AgentX on GLM-5.3-Flash with vLLM on a Kubernetes / OpenShift node with 4 free Blackwell GPUs. See README.md.
# Typical: make box prepare, then make run CONC=16 DURATION=3600 WARMUP=10 RUN_ID=full-c16-1, then make fetch-results.
# KUBE_CONTEXT and NS select the cluster and namespace (default: the current ones). NODE holds the node-local model disk.
# PREFIX names the resources (disk, pods, ConfigMap), so several setups can live in one namespace.
KUBE_CONTEXT ?=
NS ?=
NODE ?= b200-02
STORAGE_CLASS ?= lvms-b200-gpu-ib-tier1-storage
PREFIX ?= glm53
K := kubectl $(if $(KUBE_CONTEXT),--context $(KUBE_CONTEXT)) $(if $(NS),-n $(NS))

CONC ?= 16
DURATION ?= 300
WARMUP ?= 1
RUN_ID ?= smoke-c$(CONC)-1
# Host-memory KV offloading per rank in GiB (empty: off), and the pod memory that has to hold it.
OFFLOAD ?=
MEM ?= 512Gi
# Pod entry script: run.sh (one AgentX point) or forward.sh (step-time microbenchmark).
ENTRY ?= run.sh
# forward.sh only: GPUs (= TP size), and the model (small clone for smoke tests: pass REVISION= LOAD_FORMAT=dummy too).
GPUS ?= 4
MODEL ?= zai-org/GLM-5.3-Flash
REVISION ?= 46a050bec13002ad29ee795c16e61cf4fa3c4a9c
LOAD_FORMAT ?= auto
DEADLINE := $(shell echo $$(( $(DURATION) + 7200 )))

SCRIPTS := serve.sh bench.sh run.sh prepare.sh forward.sh ../../benchmarking/host_load.sh
COMMON_VARS := s/__PREFIX__/$(PREFIX)/g; s/__NODE__/$(NODE)/
RUN_VARS := s/__RUN_ID__/$(RUN_ID)/g; s/__CONC__/$(CONC)/; s/__DURATION__/$(DURATION)/; s/__WARMUP__/$(WARMUP)/; s/__OFFLOAD__/$(OFFLOAD)/
POD_VARS := s/__MEM__/$(MEM)/g; s/__GPUS__/$(GPUS)/g; s/__ENTRY__/$(ENTRY)/; s/__DEADLINE__/$(DEADLINE)/
MODEL_VARS := s|__MODEL__|$(MODEL)|; s/__REVISION__/$(REVISION)/; s/__LOAD_FORMAT__/$(LOAD_FORMAT)/

scripts:
	$(K) create configmap $(PREFIX)-scripts $(addprefix --from-file=,$(SCRIPTS)) --dry-run=client -o yaml | $(K) apply -f -

box: scripts
	sed '$(COMMON_VARS); s/__STORAGE_CLASS__/$(STORAGE_CLASS)/' box.yaml | $(K) apply -f -
	$(K) wait --for=condition=Ready pod/$(PREFIX)-box --timeout=300s

# Asynchronous: follow with `make prepare-status`.
prepare:
	$(K) exec $(PREFIX)-box -- bash -c 'mkdir -p /models/home && nohup bash /scripts/prepare.sh > /models/prepare.log 2>&1 &'

prepare-status:
	$(K) exec $(PREFIX)-box -- bash -c 'tail -n 3 /models/prepare.log; du -sh /models/hf 2> /dev/null'

run: scripts
	$(K) delete pod $(PREFIX)-run-$(RUN_ID) --ignore-not-found --wait=true
	sed '$(COMMON_VARS); $(RUN_VARS); $(POD_VARS); $(MODEL_VARS)' run.yaml | $(K) apply -f -

status:
	$(K) get pods --no-headers | grep -- '-$(RUN_ID) '; $(K) exec $(PREFIX)-box -- bash -c 'cd /models/results/$(RUN_ID) && cat startup.txt status.txt 2> /dev/null; tail -n 3 server.log; tail -n 5 client.log'

# Results without the raw per-request records, which are large.
fetch-results:
	mkdir -p results
	$(K) exec $(PREFIX)-box -- tar -C /models/results --exclude='*raw*' --exclude='*.jsonl' -czf - $(RUN_ID) | tar -C results -xzf -

RUNS ?= full-c1-1 full-c1-offload-1 full-c16-1 full-c16-offload-1 full-c24-1 full-c24-offload-1 full-c32-1 full-c32-2 full-c32-offload-1 full-c48-1

summary:
	~/.venv/bin/python summarize.py $(RUNS)

plot:
	~/.venv/bin/python plot_pareto.py $(RUNS) --reference "SGLang 0.5.22 on GB300 (SemiAnalysis InferenceX)=reference/semianalysis-sglang-fp8.csv"

clean:
	$(K) delete pod -l app=$(PREFIX)-run; $(K) delete pod $(PREFIX)-box --ignore-not-found --wait=false
