"""Measure XPress acceptance metrics, estimator state and GSM8K accuracy with
adaptive verification off and on, in the checkout that is on PYTHONPATH.

One process runs both variants sequentially: loading the target twice is the
dominant cost, and the paired outputs on identical prompts are what the
correctness comparison needs.
"""

import hashlib
import json
import os
import sys
from typing import Any

QWEN3 = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
XPRESS = ("UIUC-SSAIL/Qwen3-8B-XPress-b16", "39ab6e7ffbd0eb95b98ae799203434c98c734673")
NUM_SPECULATIVE_TOKENS = 15
VARIANTS = {"av_off": False, "av_on": True}
GSM8K_QUESTIONS = int(os.environ.get("GSM8K_QUESTIONS", "300"))


def main() -> None:
    print(f"commit={os.environ.get('VLLM_COMMIT')} version={version()}", flush=True)
    results = {name: run(av) for name, av in VARIANTS.items()}
    off, on = results["av_off"], results["av_on"]
    results["comparison"] = {
        "outputs_identical": off.pop("outputs_sha256") == on.pop("outputs_sha256"),
        "accuracy_delta": on["gsm8k_accuracy"] - off["gsm8k_accuracy"],
    }
    print("RESULT " + json.dumps(results), flush=True)


def run(enable_adaptive_verification: bool) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
    from tests.v1.e2e.spec_decode.utils import (
        compute_acceptance_len,
        compute_acceptance_rate,
        get_test_prompts,
    )

    spec: dict[str, Any] = {
        "method": "dflash",
        "model": snapshot(XPRESS),
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
    }
    if enable_adaptive_verification:
        spec["enable_adaptive_verification"] = True
    llm = LLM(
        model=snapshot(QWEN3),
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        max_num_seqs=16,
        enable_prefix_caching=False,
        speculative_config=spec,
        disable_log_stats=False,
        compilation_config={"max_cudagraph_capture_size": 128},
    )
    prompts = get_test_prompts(mm_enabled=False, num_prompts=200)
    sampling = SamplingParams(temperature=0.0, max_tokens=256, seed=0)
    chat_kwargs = {"chat_template_kwargs": {"enable_thinking": False}}
    outputs = llm.chat(prompts, sampling, **chat_kwargs)
    token_ids = [list(o.outputs[0].token_ids) for o in outputs]
    metrics = llm.get_metrics()
    counters = {
        m.name: m.value
        for m in metrics
        if m.name.startswith("vllm:spec_decode") and hasattr(m, "value")
    }
    drafts = counters["vllm:spec_decode_num_drafts"]
    result: dict[str, Any] = {
        "enable_adaptive_verification": enable_adaptive_verification,
        "num_drafts": drafts,
        "drafts_per_step": counters["vllm:spec_decode_num_draft_tokens"] / drafts,
        "acceptance_rate": compute_acceptance_rate(metrics),
        "acceptance_len": compute_acceptance_len(metrics),
        "num_output_tokens": sum(map(len, token_ids)),
        "outputs_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()[:16],
        "workers": llm.collective_rpc(speculator_state),
    }
    gsm8k = evaluate_gsm8k_offline(llm, num_questions=GSM8K_QUESTIONS, max_tokens=256)
    result["gsm8k_accuracy"] = gsm8k["accuracy"]
    result["estimator_after_eval"] = llm.collective_rpc(speculator_state)
    del llm
    free_memory()
    return result


def speculator_state(worker) -> dict[str, Any]:
    speculator = worker.model_runner.speculator
    estimator = speculator.acceptance_estimator
    if estimator is None:
        return {"speculator": type(speculator).__name__, "estimator": None}
    features = estimator.features.float()
    return {
        "speculator": type(speculator).__name__,
        "estimator": {
            "refits": estimator._refits,
            "features_nonzero": int((features != 0).sum()),
            "features_mean": float(features.mean()),
            "predictions_nonzero": int((estimator.predictions != 0).sum()),
            "predictions_mean": float(estimator.predictions.mean()),
            "slope": float(estimator.slope),
            "intercepts": [round(float(x), 3) for x in estimator.intercepts],
        },
    }


def version() -> str:
    import vllm

    return vllm.__version__


def free_memory() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def snapshot(repo: tuple[str, str]) -> str:
    name, revision = repo
    return f"{os.environ['HF_HUB_CACHE']}/models--{name.replace('/', '--')}/snapshots/{revision}"


if __name__ == "__main__":
    main()
