import itertools
import os

port_source = itertools.count(start=15000, step=1)

temps = [0.0]
tp_sizes = [1, 2]

os.makedirs("./generated", exist_ok=True)

with open("./throughput-sd-template.slurm", "rt") as f:
    tsd_template = f.read()

method_and_model = [
    ("draft_model", "Qwen/Qwen3-0.6B"),
    ("draft_model", "Qwen/Qwen3-1.7B"),
    ("eagle3", "RedHatAI/Qwen3-32B-speculator.eagle3"),
]

combinations = itertools.product(temps, tp_sizes, method_and_model)
for temp, tp_size, (method, model) in combinations:
    model_short = model.split("/")[-1]
    jobname = f"vllm-throughput-sd-{method}-{model_short}-t{temp:.1f}-tp{tp_size}"
    filled = tsd_template.format(
        JOB_NAME=jobname,
        TEMPERATURE=temp,
        VLLM_PORT=next(port_source),
        N_GPUS=tp_size,
        SPECULATIVE_METHOD=method,
        SPECULATIVE_MODEL=model,
    )
    with open(f"./generated/{jobname}.slurm", "wt") as f:
        f.write(filled)

with open("./throughput-nosd-template.slurm", "rt") as f:
    tnsd_template = f.read()

combinations = itertools.product(temps, tp_sizes)
for temp, tp_size in combinations:
    jobname = f"vllm-throughput-nosd-t{temp:.1f}-tp{tp_size}"
    filled = tnsd_template.format(
        JOB_NAME=jobname,
        TEMPERATURE=temp,
        VLLM_PORT=next(port_source),
        N_GPUS=tp_size,
    )
    with open(f"./generated/{jobname}.slurm", "wt") as f:
        f.write(filled)

print("Done creating slurm files")
