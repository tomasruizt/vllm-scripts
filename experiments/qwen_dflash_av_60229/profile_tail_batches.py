"""Trace native AV dummy tails and record the actual backend metadata."""

import json
import os
from pathlib import Path

import torch

import profile_raw_cost_curve  # Install the raw, pre-clamp startup hook.


def main() -> None:
    from vllm import LLM

    llm = LLM(
        model=os.environ["MODEL"], language_model_only=True,
        max_model_len=16384, max_num_seqs=512, max_num_batched_tokens=16384,
        gpu_memory_utilization=0.92, enable_prefix_caching=False,
        worker_extension_cls="profile_tail_batches.TailWorker",
        speculative_config=dict(
            method="dflash", model="z-lab/Qwen3.5-35B-A3B-DFlash",
            revision="52cb554b4995dede3e2e1bdb129cdb1f3529332b",
            num_speculative_tokens=7, draft_sample_method="probabilistic",
            enable_adaptive_verification=True,
        ),
    )
    print(llm.collective_rpc("trace_tail_batches"), flush=True)
    llm.llm_engine.engine_core.shutdown()
    print("TAIL_PROFILE_DONE", flush=True)


class TailWorker:
    def trace_tail_batches(self) -> str:
        runner = self.model_runner
        root = Path(os.environ["TAIL_OUTPUT"])
        root.mkdir(parents=True, exist_ok=True)
        mixed = os.environ["TAIL_ARM"] == "mixed"
        rows = []
        for policy in (("mixed", "prefill") if mixed else ("slot_only",)):
            for tokens in (4096, 8192, 16384):
                row = trace_batch(runner, root, policy, tokens)
                rows.append(row)
                (root / "batches.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)
        return str(root)


def trace_batch(runner: object, root: Path, policy: str, tokens: int) -> dict:
    metadata = []
    prepare = runner.model_state.prepare_attn
    max_num_reqs = runner.max_num_reqs

    def capture(batch: object, mode: object, *args: object, **kwargs: object) -> dict:
        result = prepare(batch, mode, *args, **kwargs)
        metadata.append(describe_batch(batch, mode, result))
        return result

    runner.model_state.prepare_attn = capture
    if policy == "prefill":
        runner.max_num_reqs = 1
    try:
        with runner.step_timing.collect() as timings:
            for _ in range(3):
                runner._dummy_run(tokens, context_len=1024, profile_mixed=policy != "slot_only")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            runner._dummy_run(tokens, context_len=1024, profile_mixed=policy != "slot_only")
            torch.cuda.synchronize()
        label = f"{policy}-{tokens}"
        profiler.export_chrome_trace(str(root / f"{label}-trace.json"))
        (root / f"{label}-metadata.json").write_text(json.dumps(metadata, indent=2))
        return dict(policy=policy, tokens=tokens,
                    target_ms=[sample.forward_ms for sample in timings],
                    drafter_ms=[sample.drafter_ms for sample in timings],
                    batch=metadata[-1])
    finally:
        runner.model_state.prepare_attn = prepare
        runner.max_num_reqs = max_num_reqs


def describe_batch(batch: object, mode: object, metadata: dict) -> dict:
    prefills = batch.is_prefilling_np
    decode_lens = batch.num_scheduled_tokens[~prefills]
    prefill_lens = batch.num_scheduled_tokens[prefills]
    gdn = [value for value in metadata.values() if hasattr(value, "num_spec_decodes")]
    first = gdn[0] if gdn else None
    result = dict(
        requests=batch.num_reqs, tokens=batch.num_tokens,
        padded_tokens=batch.num_tokens_after_padding, mode=str(mode),
        decode_requests=int((~prefills).sum()), prefill_requests=int(prefills.sum()),
        decode_tokens=int(decode_lens.sum()), prefill_tokens=int(prefill_lens.sum()),
        max_decode_query=int(decode_lens.max()) if decode_lens.size else 0,
        prefill_query_lengths=prefill_lens.tolist(), gdn_layers=len(gdn),
    )
    if first is not None:
        result["gdn"] = {key: int(getattr(first, key)) for key in (
            "num_spec_decodes", "num_spec_decode_tokens", "num_prefills", "num_prefill_tokens",
        )}
        indices = first.spec_state_indices_tensor
        accepted = first.num_accepted_tokens
        if indices is not None and accepted is not None and first.num_spec_decodes:
            count = first.num_spec_decodes
            sources = indices[:count].gather(1, accepted[:count, None].long() - 1)
            result["gdn"]["null_sources"] = int((sources <= 0).sum())
    return result


if __name__ == "__main__":
    main()
