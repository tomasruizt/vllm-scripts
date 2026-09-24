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
* You dont need to explain metrics like ITL, TPOT, etc. When writing summaries, assume a technical audience that is familiar with the domain.
* When available, use hf models from `RedHatAI`, e.g. speculators.
* If you are benchmarking a model with multiple runs, try to spin up the vllm server only once, since starting the server is what takes often most time.

custom commands:
* vllm-install: install from precompiled wheels
* set-cuda-visible: set CUDA_VISIBLE_DEVICES env var depending on the GPUs reserved using canhazgpu.
* git aliases
