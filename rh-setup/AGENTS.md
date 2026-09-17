* Be concise in your answers
* Latex doesn't render well in the terminal. Use an alternative.
* Use the python env in ~/.venv/bin/python. Use `uv` to install python 3.12.
* When creating PRs that are meant only for reviewing changes between two commits, don't mention the PRs or issues, because they become polluted with these references.
* structure code top-down. i.e. top in the file is high-level code/policy, while below is lower-level code that is used by the higher-level code.
* to use GPUs, reserve them over canhazgpu (https://github.com/russellb/canhazgpu).

custom commands:
* vllm-install: install from precompiled wheels
* set-cuda-visible: set CUDA_VISIBLE_DEVICES env var depending on the GPUs reserved using canhazgpu.
* git aliases