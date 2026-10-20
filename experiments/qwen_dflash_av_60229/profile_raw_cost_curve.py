"""Record native startup cost curves before monotonic clamping."""

import json
import os
from pathlib import Path

from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager

_set_cost_curves = AdaptiveVerificationManager.set_cost_curves


def record_cost_curves(self: object, draft_curve: list, verify_curve: list) -> None:
    Path(os.environ["RAW_CURVE_OUTPUT"]).write_text(json.dumps(
        dict(draft_curve=draft_curve, verify_curve=verify_curve), indent=2))
    _set_cost_curves(self, draft_curve, verify_curve)


AdaptiveVerificationManager.set_cost_curves = record_cost_curves


def main() -> None:
    from vllm import LLM

    llm = LLM(
        model=os.environ["MODEL"], language_model_only=True,
        max_model_len=16384, max_num_seqs=512, max_num_batched_tokens=16384,
        gpu_memory_utilization=0.92, enable_prefix_caching=False,
        speculative_config=dict(
            method="dflash", model="z-lab/Qwen3.5-35B-A3B-DFlash",
            revision="52cb554b4995dede3e2e1bdb129cdb1f3529332b",
            num_speculative_tokens=7, draft_sample_method="probabilistic",
            enable_adaptive_verification=True,
        ),
    )
    assert Path(os.environ["RAW_CURVE_OUTPUT"]).exists()
    llm.llm_engine.engine_core.shutdown()
    print("RAW_CURVE_DONE", flush=True)


if __name__ == "__main__":
    main()
