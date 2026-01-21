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

dataset_and_concurrencies = [
    ("likaixin/InstructCoder", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
    ("philschmid/mt-bench", [1, 2, 4, 8, 16, 32, 64, 80]),
]

combinations = itertools.product(
    temps, tp_sizes, method_and_model, dataset_and_concurrencies
)
for temp, tp_size, (method, model), (dataset, concurrencies) in combinations:
    model_short = model.split("/")[-1]
    dataset_short = dataset.split("/")[-1]
    jobname = f"vllm-throughput-{dataset_short}-sd-{method}-{model_short}-t{temp:.1f}-tp{tp_size}"
    filled = tsd_template.format(
        JOB_NAME=jobname,
        TEMPERATURE=temp,
        VLLM_PORT=next(port_source),
        N_GPUS=tp_size,
        SPECULATIVE_METHOD=method,
        SPECULATIVE_MODEL=model,
        DATASET=dataset,
        CONCURRENCIES=concurrencies,
    )
    with open(f"./generated/{jobname}.slurm", "wt") as f:
        f.write(filled)

with open("./throughput-nosd-template.slurm", "rt") as f:
    tnsd_template = f.read()

combinations = itertools.product(temps, tp_sizes, dataset_and_concurrencies)
for temp, tp_size, (dataset, concurrencies) in combinations:
    dataset_short = dataset.split("/")[-1]
    jobname = f"vllm-throughput-{dataset_short}-nosd-t{temp:.1f}-tp{tp_size}"
    filled = tnsd_template.format(
        JOB_NAME=jobname,
        TEMPERATURE=temp,
        VLLM_PORT=next(port_source),
        N_GPUS=tp_size,
        DATASET=dataset,
        CONCURRENCIES=concurrencies,
    )
    with open(f"./generated/{jobname}.slurm", "wt") as f:
        f.write(filled)

print("Done creating slurm files")
