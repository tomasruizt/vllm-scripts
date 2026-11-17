"""Per-step GPU time breakdown of DSpine decoding (target, verify+sample, draft).

Runs Qwen3.8-27B + tencent/Qwen3.8-27B-DSpine-b16 offline (in-process engine),
submits `c` GSM8K prompts at once for each concurrency `c`, and times the steps
where all `c` requests are running with CUDA events around the model runner's
`execute_model`, `sample_tokens` and the speculator's `propose`.
"""

import argparse
import json
import os
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

DRAFT = "tencent/Qwen3.8-27B-DSpine-b16"


def main() -> None:
    args = parse_args()
    llm = make_llm(args)
    prompts = gsm8k_prompts()
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    chat = lambda ps: llm.chat(  # noqa: E731
        ps, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False
    )
    chat(prompts[: args.concurrency[-1]])  # warmup
    llm.collective_rpc(install_timers)
    offset = args.concurrency[-1]
    if args.nsys:  # with `nsys profile --capture-range=cudaProfilerApi`
        torch.cuda.profiler.start()
    with open(args.out, "w") as f:
        for c in args.concurrency:
            for rnd in range(args.rounds):
                batch = [prompts[i % len(prompts)] for i in range(offset, offset + c)]
                offset += c
                f.write(json.dumps(run_round(llm, chat, batch, c, rnd)) + "\n")
                f.flush()
    if args.nsys:
        torch.cuda.profiler.stop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--num-spec-tokens", type=int, default=15)
    p.add_argument("--out", default="steps.jsonl")
    p.add_argument("--nsys", action="store_true")
    p.add_argument("--target", default="Qwen/Qwen3.8-27B")
    p.add_argument("--target-revision", default="37561816e66b5b9153b1b64c10968c18c1e6f4a0")
    p.add_argument("--draft-revision", default="260a6c23070ca873ff1a5f33b7119ac3f6a245ee")
    p.add_argument("--attention-config", type=json.loads, default=None)
    return p.parse_args()


def make_llm(args: argparse.Namespace) -> LLM:
    return LLM(
        model=args.target,
        revision=args.target_revision,
        dtype="bfloat16",
        tensor_parallel_size=1,
        language_model_only=True,
        max_model_len=4096,
        max_num_seqs=max(args.concurrency),
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        generation_config="vllm",
        seed=0,
        disable_log_stats=False,
        attention_config=args.attention_config,
        speculative_config={
            "model": DRAFT,
            "revision": args.draft_revision,
            "method": "dflash",
            "num_speculative_tokens": args.num_spec_tokens,
            "enable_adaptive_verification": False,
        },
    )


def gsm8k_prompts() -> list[list[dict]]:
    rows = load_dataset("openai/gsm8k", "main", split="test")
    return [[{"role": "user", "content": r["question"]}] for r in rows]


def run_round(llm: LLM, chat, batch: list, c: int, rnd: int) -> dict:
    before = spec_counters(llm)
    llm.collective_rpc(set_timing, args=(True,))
    start = time.perf_counter()
    outputs = chat(batch)
    wall = time.perf_counter() - start
    steps = llm.collective_rpc(set_timing, args=(False,))[0]
    after = spec_counters(llm)
    drafts = after["drafts"] - before["drafts"]
    accepted = after["accepted"] - before["accepted"]
    return dict(
        concurrency=c, round=rnd, wall_s=wall,
        output_tokens=sum(len(o.outputs[0].token_ids) for o in outputs),
        acceptance_len=1 + accepted / max(drafts, 1),
        steps=steps,
    )


def spec_counters(llm: LLM) -> dict:
    values = {m.name: getattr(m, "value", None) for m in llm.get_metrics()}
    return dict(
        drafts=values.get("vllm:spec_decode_num_drafts") or 0,
        accepted=values.get("vllm:spec_decode_num_accepted_tokens") or 0,
    )


# --- Run in the worker (in-process engine) ---


def install_timers(worker) -> None:
    runner = worker.model_runner
    runner._dspine_log, runner._dspine_on = [], [False]

    def wrap(obj, name: str, tag: str) -> None:
        orig = getattr(obj, name)

        def timed(*a, **k):
            if not runner._dspine_on[0]:
                return orig(*a, **k)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            nreq = None
            if tag == "execute":
                nreq = (len(a[0].num_scheduled_tokens), a[0].total_num_scheduled_tokens)
            start.record()
            torch.cuda.nvtx.range_push(tag)
            out = orig(*a, **k)
            torch.cuda.nvtx.range_pop()
            end.record()
            runner._dspine_log.append((tag, start, end, nreq))
            return out

        setattr(obj, name, timed)

    wrap(runner, "execute_model", "execute")
    wrap(runner, "sample_tokens", "sample")
    wrap(runner.speculator, "propose", "draft")


def set_timing(worker, on: bool) -> list[dict]:
    runner = worker.model_runner
    runner._dspine_on[0] = on
    if on:
        return []
    log, runner._dspine_log = runner._dspine_log, []
    torch.cuda.synchronize()
    steps: list[dict] = []
    for tag, start, end, nreq in log:
        if tag == "execute":
            if steps:
                steps[-1]["period_ms"] = steps[-1].pop("_start").elapsed_time(start)
            steps.append(dict(num_reqs=nreq[0], num_tokens=nreq[1], _start=start))
        if steps:
            steps[-1][f"{tag}_ms"] = start.elapsed_time(end)
    if steps:
        steps[-1].pop("_start")
    return [s for s in steps if "period_ms" in s]


if __name__ == "__main__":
    main()
