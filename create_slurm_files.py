import itertools
import os

port_source = itertools.count(start=15000, step=1)

with open("./acceptance-len-template.slurm", "rt") as f:
    al_template = f.read()

temps = [0.0]
tp_sizes = [1, 2]

for temp, tp_size in itertools.product(temps, tp_sizes):
    jobname = f"vllm-acceptance-len-t{temp:.1f}-tp{tp_size}"
    filled = al_template.format(
        JOB_NAME=jobname,
        TEMPERATURE=temp,
        N_GPUS=tp_size,
    )
    os.makedirs("./generated", exist_ok=True)
    with open(f"./generated/{jobname}.slurm", "wt") as f:
        f.write(filled)

with open("./throughput-sd-template.slurm", "rt") as f:
    tsd_template = f.read()

combinations = itertools.product(temps, tp_sizes)
for (temp, tp_size) in combinations:
    jobname = f"vllm-throughput-sd-t{temp:.1f}-tp{tp_size}"
    filled = tsd_template.format(
        JOB_NAME=jobname,
        TEMPERATURE=temp,
        VLLM_PORT=next(port_source),
        N_GPUS=tp_size,
    )
    with open(f"./generated/{jobname}.slurm", "wt") as f:
        f.write(filled)
        
with open("./throughput-nosd-template.slurm", "rt") as f:
    tnsd_template = f.read()

combinations = itertools.product(temps, tp_sizes)
for (temp, tp_size) in combinations:
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