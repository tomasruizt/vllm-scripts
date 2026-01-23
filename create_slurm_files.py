import itertools
import os

port_source = itertools.count(start=15000, step=1)

temps = [0.0]
tp_sizes = [1]  # , 2]
all_n_spec_toks = [1, 2, 3, 4, 5, 6]


def dump_slurm_files(sd_template: str, nosd_template: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    with open(sd_template, "rt") as f:
        tsd_template = f.read()

    tgtmodel_method_and_dftmodel = [
        ("Qwen/Qwen3-32B", "draft_model", "Qwen/Qwen3-1.7B"),
        ("Qwen/Qwen3-32B", "draft_model", "Qwen/Qwen3-4B"),
        ("Qwen/Qwen3-32B", "eagle3", "RedHatAI/Qwen3-32B-speculator.eagle3"),
    ]

    dataset_concurrencies_num_prompts = [
        ("philschmid/mt-bench", "1 2 4 8 16 32 50 80", "80", ""),
        (
            "likaixin/InstructCoder",
            "1 2 4 8 16 32 64 128 256",
            # dynamic num prompts: MAX(MAX_CONCURRENCY * 10, 50)
            "\$(( \$MAX_CONCURRENCY * 10 ))",
            "if [ \$NUM_PROMPTS -lt 50 ]; then NUM_PROMPTS=50 fi",
        ),
    ]

    combinations = itertools.product(
        temps,
        tp_sizes,
        tgtmodel_method_and_dftmodel,
        dataset_concurrencies_num_prompts,
        all_n_spec_toks,
    )
    for temp, tp_size, (tgtmodel, method, dftmodel), (
        dataset,
        concurrencies,
        num_prompts,
        num_prompts_extra,
    ), n_spec_toks in combinations:
        model_short = dftmodel.split("/")[-1]
        dataset_short = dataset.split("/")[-1]
        jobname = f"vllm-throughput-{dataset_short}-sd-{method}-{model_short}-k{n_spec_toks}-t{temp:.1f}-tp{tp_size}"
        filled = tsd_template.format(
            JOB_NAME=jobname,
            TEMPERATURE=temp,
            VLLM_PORT=next(port_source),
            N_GPUS=tp_size,
            TARGET_MODEL=tgtmodel,
            SPECULATIVE_METHOD=method,
            SPECULATIVE_MODEL=dftmodel,
            DATASET=dataset,
            CONCURRENCIES=concurrencies,
            NUM_PROMPTS=num_prompts,
            NUM_PROMPTS_EXTRA=num_prompts_extra,
            N_SPEC_TOKS=n_spec_toks,
        )
        with open(out_dir + f"/{jobname}.slurm", "wt") as f:
            f.write(filled)

    with open(nosd_template, "rt") as f:
        tnsd_template = f.read()

    combinations = itertools.product(
        temps, tp_sizes, tgtmodel_method_and_dftmodel, dataset_concurrencies_num_prompts
    )
    for temp, tp_size, (tgtmodel, _, _), (
        dataset,
        concurrencies,
        num_prompts,
        num_prompts_extra,
    ) in combinations:
        dataset_short = dataset.split("/")[-1]
        jobname = f"vllm-throughput-{dataset_short}-nosd-t{temp:.1f}-tp{tp_size}"
        filled = tnsd_template.format(
            JOB_NAME=jobname,
            TEMPERATURE=temp,
            VLLM_PORT=next(port_source),
            N_GPUS=tp_size,
            TARGET_MODEL=tgtmodel,
            DATASET=dataset,
            CONCURRENCIES=concurrencies,
            NUM_PROMPTS=num_prompts,
            NUM_PROMPTS_EXTRA=num_prompts_extra,
        )
        with open(out_dir + f"/{jobname}.slurm", "wt") as f:
            f.write(filled)

    print("Created slurm files in", out_dir)


if __name__ == "__main__":
    sd_template = "./throughput-sd-template.slurm"
    nosd_template = "./throughput-nosd-template.slurm"
    out_dir = "./generated-lrz"
    dump_slurm_files(sd_template, nosd_template, out_dir)

    # KISSKI
    sd_template = "./slurm-kisski/throughput-sd-template.slurm"
    nosd_template = "./slurm-kisski/throughput-nosd-template.slurm"
    out_dir = "./generated-kisski"
    dump_slurm_files(sd_template, nosd_template, out_dir)
