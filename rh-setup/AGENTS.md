* Be concise in your answers
* Keep markdown reports / summaries VERY concise and to the point.
* Configure Codex in `~/.codex/config.toml` with `approval_policy = "on-request"` and `approvals_reviewer = "auto_review"` to route eligible command approvals to an automatic reviewer.
* Dont use python to make simple file edits. Please use your built in `apply_patch` command, or equivalent, instead.
* Latex doesn't render well in the terminal. Use an alternative.
* Use the python env in ~/.venv/bin/python. Use `uv` to install python 3.12.
* When creating PRs that are meant only for reviewing changes between two commits, don't mention the PRs or issues, because they become polluted with these references.
* structure code top-down. i.e. top in the file is high-level code/policy, while below is lower-level code that is used by the higher-level code.
* to use GPUs, reserve them over canhazgpu (https://github.com/russellb/canhazgpu).
* when writing md files, dont break lines unless there is a good reason, since e.g. on Github your lines will be wrapped. Let the IDE / website take care of rendering, so you don't have to. I suggest only breaking lines after a dot, or similar.
* in terms of writing stlye, don't use em-dashes, or ";", because I dont use them either. Use commas, dots, parenthesis and colons instead.
* Never rewrite the style of my prose, unless I explictily ask you to.
* You dont need to explain metrics like ITL, TPOT, etc. When writing summaries, assume a technical audience that is familiar with the domain.
* When available, use hf models from `RedHatAI`, e.g. speculators.
* Before downloading a hf model, check if the system has a mount where all users can share the model weights, and set it as a download location. Sometimes that path is not writeable, so check first.
* If you are benchmarking a model with multiple runs, try to spin up the vllm server only once, since starting the server is what takes often most time. However, in these cases we most usually want to disable prefix-caching, because otherwise the second and later runs would be artificially faster.
* For perf benchmarks prefer to use standard tooling. This includes e.g. `vllm bench sweep serve`, or ai-perf or similar things. These capture important metrics like TPOT, ITL, etc. If you are writing custom code and forget to capture metrics, we are wasting our fucking time.
* Pass `--disable-uvicorn-access-log` in general to reduce the log verbosity.
* Use the github cli `gh` when possible
* When making code changes, try to reduce the git diff, so reviews are easier for humans.
* In general, its better to use `chg run` rather than manual GPU reservations, unless strictly necessary.
* When doing sweeps over concurrencies and comparing methods, we need to be mindful of the runtime. In particular, running a workload at c8 is going to be much slower than at c256. We need to size the workload so as to spend ~30s per measurement (independent of batch size). Because we are running this accross K different c levels, and N different n_runs. Perhaps we can randomly subsample the workload for smaller c, and not compare absolute runtimes. We need to be smart about this!!! Think about the design before spending many hours running stuff!
* launch workloads on remote machines in asynchronous mode, so your main loop is never blocked, and you can continue to receive instructions and continue to perform other tasks.
* Keep your diffs small, the code vertically compact, and indentation small. Use functions to get rid of indentation. Use variables to avoid large multi-line calls that. Try to put the bodies of loops in a function to make it ovious what comes out of the loop. Use pure functions when possible. Staircase-shaped code that is with context managers, loops, list and object creations, etc. should be generally avoided. Use type hints whenever you can.
* in unit tests dont set monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1"), this should be set by the caller

custom commands:
* vllm-install: install from precompiled wheels
* set-cuda-visible: set CUDA_VISIBLE_DEVICES env var depending on the GPUs reserved using canhazgpu.
* git aliases
l