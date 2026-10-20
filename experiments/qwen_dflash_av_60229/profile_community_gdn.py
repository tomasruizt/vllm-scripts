"""Record community-native calibration and GDN metadata on both arms."""

import json
import os
from pathlib import Path

import profile_raw_cost_curve  # noqa: F401, install the native raw-curve hook.
from profile_tail_batches import describe_batch


def main() -> None:
    from vllm import LLM

    llm = LLM(
        model=os.environ["MODEL"], language_model_only=True,
        max_model_len=16384, max_num_seqs=512, max_num_batched_tokens=16384,
        gpu_memory_utilization=0.92, enable_prefix_caching=False,
        worker_extension_cls="profile_community_gdn.CommunityGDNWorker",
        speculative_config=dict(
            method="dflash", model="z-lab/Qwen3.5-35B-A3B-DFlash",
            revision="52cb554b4995dede3e2e1bdb129cdb1f3529332b",
            num_speculative_tokens=7, draft_sample_method="probabilistic",
            enable_adaptive_verification=True,
        ),
    )
    print(llm.collective_rpc("record_gdn_batches"), flush=True)
    llm.llm_engine.engine_core.shutdown()
    print("COMMUNITY_ARM_DONE", flush=True)


class CommunityGDNWorker:
    def record_gdn_batches(self) -> str:
        runner = self.model_runner
        prepare = runner.model_state.prepare_attn
        rows = []

        def capture(batch: object, mode: object, *args: object, **kwargs: object) -> dict:
            metadata = prepare(batch, mode, *args, **kwargs)
            rows.append(describe_batch(batch, mode, metadata))
            return metadata

        runner.model_state.prepare_attn = capture
        try:
            for tokens in (512, 1024, 4096, 8192, 16384):
                split = runner.adaptive_verification.profile_batch_split(tokens)
                runner._dummy_run(tokens, context_len=1024, num_scheduled_tokens_per_req=split)
        finally:
            runner.model_state.prepare_attn = prepare
        output = Path(os.environ["RAW_CURVE_OUTPUT"]).with_suffix(".batches.json")
        output.write_text(json.dumps(rows, indent=2))
        return str(output)


if __name__ == "__main__":
    main()
