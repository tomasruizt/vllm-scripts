SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
REMOTE ?= RH-nm-frk-h100-01-preserve
RUN_ID ?= dspine-profile-20261113
REMOTE_ROOT ?= runs
SEQUENCES ?= ref,fused;fused,ref
PROFILE_ARGS ?=
SSH := ssh -T -o BatchMode=yes -o ConnectTimeout=10

.PHONY: launch-profile watch fetch-results
launch-profile:
	$(SSH) $(REMOTE) 'mkdir "$(REMOTE_ROOT)/$(RUN_ID)"'  # fails if the run exists: pick a new RUN_ID
	scp -q profile_steps.py launch_profile.sh $(REMOTE):"$(REMOTE_ROOT)/$(RUN_ID)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)" && ROOT=$$PWD SEQUENCES="$(SEQUENCES)" PROFILE_ARGS="$(PROFILE_ARGS)" TREE="$(TREE)" HF_HOME_DIR="$(HF_HOME_DIR)" setsid nohup bash launch_profile.sh > launch.log 2>&1 < /dev/null & echo launched'

watch:
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)"; timeout 7200 bash -c "until test -f launch-exit-code.txt; do sleep 10; done"'
	$(MAKE) fetch-results

fetch-results:  # excludes nsys reports (large)
	mkdir -p results
	$(SSH) $(REMOTE) 'tar czf - --exclude="*.nsys-rep" --exclude="*.sqlite" -C "$(REMOTE_ROOT)" "$(RUN_ID)"' | tar xzf - -C results

VARIANT ?= ref
.PHONY: launch-nsys
launch-nsys:
	$(SSH) $(REMOTE) 'mkdir "$(REMOTE_ROOT)/$(RUN_ID)"'
	scp -q profile_steps.py launch_nsys.sh $(REMOTE):"$(REMOTE_ROOT)/$(RUN_ID)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)" && ROOT=$$PWD VARIANT="$(VARIANT)" TREE="$(TREE)" PROFILE_ARGS="$(PROFILE_ARGS)" setsid nohup bash launch_nsys.sh > launch.log 2>&1 < /dev/null & echo launched'

.PHONY: launch-equivalence
launch-equivalence:
	$(SSH) $(REMOTE) 'mkdir "$(REMOTE_ROOT)/$(RUN_ID)"'
	scp -q equivalence.py launch_equivalence.sh $(REMOTE):"$(REMOTE_ROOT)/$(RUN_ID)/"
	$(SSH) $(REMOTE) 'cd "$(REMOTE_ROOT)/$(RUN_ID)" && ROOT=$$PWD setsid nohup bash launch_equivalence.sh > launch.log 2>&1 < /dev/null & echo launched'
