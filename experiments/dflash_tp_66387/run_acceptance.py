"""Measure DFlash acceptance length for one tensor-parallel size (issue #66387).

Runs a fixed offline workload with greedy sampling and reports the spec-decode
stats that the issue quotes: mean acceptance length and the per-position
acceptance rate vector.
"""

import argparse
import json
import logging
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-spec-tokens", type=int, default=15)
    parser.add_argument("--prompts", required=True, help="AIPerf SPEED-Bench jsonl")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="Repeat each user message this many times to grow the prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--concat-prompts",
        type=int,
        default=1,
        help="Join this many source prompts into each long prompt",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--draft-sample-method",
        default=None,
        help="greedy or probabilistic, omitted by default",
    )
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--cudagraph-mode", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-spec", action="store_true", help="Target model only")
    parser.add_argument("--dump-tokens", default=None, help="Write token ids per prompt")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_prompts(
    path: str, num_prompts: int, prompt_repeat: int, concat_prompts: int
) -> list[list[dict]]:
    """Build chat conversations from the source jsonl.

    ``concat_prompts`` joins successive source prompts into one long prompt, so
    a long-context cell still uses distinct natural text instead of repetition.
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    contents = [row["messages"][-1]["content"] for row in rows]
    group = concat_prompts * prompt_repeat
    conversations = [
        [{"role": "user", "content": "\n\n".join(contents[i : i + group])}]
        for i in range(0, group * num_prompts, group)
    ]
    return conversations


def apply_chat_template(tokenizer, conversations: list[list[dict]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in conversations
    ]


class SpecMetricsCollector(logging.Handler):
    """Collect the formatted 'SpecDecoding metrics' log lines."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "SpecDecoding metrics" in message:
            self.lines.append(message)

    def last(self) -> dict | None:
        if not self.lines:
            return None
        return parse_metrics_line(self.lines[-1])


def parse_metrics_line(line: str) -> dict:
    mean = line.split("Mean acceptance length: ")[1].split(",")[0]
    rates = line.split("Per-position acceptance rate: ")[1].split(", Avg Draft")
    return {
        "mean_acceptance_length": float(mean),
        "per_position_acceptance": [float(x) for x in rates[0].split(",")],
        "raw": line,
    }


def build_llm(args: argparse.Namespace):
    from vllm import LLM

    speculative_config = {
        "method": "dflash",
        "model": args.draft,
        "num_speculative_tokens": args.num_spec_tokens,
    }
    if args.draft_sample_method is not None:
        speculative_config["draft_sample_method"] = args.draft_sample_method
    kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": 0.85,
        "disable_log_stats": False,
        "enforce_eager": args.enforce_eager,
        "language_model_only": args.language_model_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if not args.no_spec:
        kwargs["speculative_config"] = speculative_config
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.cudagraph_mode is not None:
        kwargs["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    return LLM(**kwargs)


def main() -> None:
    args = parse_args()
    os.environ["VLLM_LOG_STATS_INTERVAL"] = "100000"

    import vllm
    from vllm import SamplingParams

    print("VLLM", vllm.__version__, vllm.__file__, flush=True)
    llm = build_llm(args)
    prompts = apply_chat_template(
        llm.get_tokenizer(),
        load_prompts(
            args.prompts,
            args.num_prompts,
            args.prompt_repeat,
            args.concat_prompts,
        ),
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    outputs = llm.generate(prompts, sampling_params)

    if args.dump_tokens:
        with open(args.dump_tokens, "w") as f:
            for i, out in enumerate(outputs):
                row = {
                    "i": i,
                    "prompt_token_ids": list(out.prompt_token_ids),
                    "token_ids": list(out.outputs[0].token_ids),
                }
                f.write(json.dumps(row) + "\n")

    # vLLM installs its own logging handlers at import, so attach after that.
    collector = SpecMetricsCollector()
    logging.getLogger("vllm").addHandler(collector)
    llm.llm_engine.do_log_stats()
    logging.getLogger("vllm").removeHandler(collector)

    result = collector.last() or {"mean_acceptance_length": None}
    result.update(
        {
            "vllm_version": vllm.__version__,
            "vllm_path": vllm.__file__,
            "tp": args.tp,
            "num_spec_tokens": args.num_spec_tokens,
            "num_prompts": len(prompts),
            "prompt_repeat": args.prompt_repeat,
            "prompt_tokens": [len(o.prompt_token_ids) for o in outputs],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "draft_sample_method": args.draft_sample_method,
            "concat_prompts": args.concat_prompts,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "cudagraph_mode": args.cudagraph_mode,
            "enforce_eager": args.enforce_eager,
            "generated_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
            "log_lines": collector.lines,
        }
    )
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT", json.dumps(result["mean_acceptance_length"]), result.get("per_position_acceptance"))


if __name__ == "__main__":
    main()
