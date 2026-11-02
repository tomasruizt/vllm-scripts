"""Smoke test adaptive verification with a trained confidence head vs the online estimator (DSpark, XPress)."""
import json
import os
import sys
from typing import Any

DSV4 = ("inference-optimization/DeepSeek-V4-Flash-DSpark-2.1B-A1.9B-small", "bbddd703cacc556881da865ecb67663454e943ab")
QWEN3 = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
XPRESS = ("UIUC-SSAIL/Qwen3-8B-XPress-b16", "39ab6e7ffbd0eb95b98ae799203434c98c734673")
DSPARK = {"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic"}
AV = {"enable_adaptive_verification": True}
ARGMAX = AV | {"draft_sample_method": "greedy", "use_local_argmax_reduction": True}
# name -> (target, speculative config, load format). DSpark runs dummy weights, XPress real ones.
VARIANTS: dict[str, tuple[tuple[str, str], dict[str, Any], str]] = {
    "auto": (DSV4, DSPARK | AV, "dummy"),
    "trained": (DSV4, DSPARK | AV | {"confidence_head": "trained"}, "dummy"),
    "online": (DSV4, DSPARK | AV | {"confidence_head": "online_estimator"}, "dummy"),
    "no_av": (DSV4, DSPARK | {"confidence_head": "online_estimator"}, "dummy"),  # must fail at startup
    "online_argmax": (DSV4, DSPARK | ARGMAX | {"confidence_head": "online_estimator"}, "dummy"),  # must fail in SpeculativeConfig
    "trained_argmax": (DSV4, DSPARK | ARGMAX | {"confidence_head": "trained"}, "dummy"),  # passes SpeculativeConfig
    "xpress": (QWEN3, {"method": "dflash", "model": XPRESS, "num_speculative_tokens": 15}, "auto"),
    "xpress_av": (QWEN3, {"method": "dflash", "model": XPRESS, "num_speculative_tokens": 15} | AV, "auto"),
}


def main() -> None:
    variant = sys.argv[1]
    target, spec, load_format = VARIANTS[variant]
    result = {"variant": variant, "target": target[0], "spec": spec, "load_format": load_format}
    print("RESULT " + json.dumps(result | run(target, spec, load_format)), flush=True)


def run(target: tuple[str, str], spec: dict[str, Any], load_format: str) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    try:
        llm = LLM(**engine_args(target, spec, load_format))
    except Exception as error:
        return {"startup_error": repr(error)[:1000]}
    prompts = [f"Write a story about the number {i}." for i in range(16)]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=256, ignore_eos=True))
    num_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    return {"num_output_tokens": num_tokens, "spec_decode": spec_decode_counters(llm),
            "workers": llm.collective_rpc(speculator_state)}


def engine_args(target: tuple[str, str], spec: dict[str, Any], load_format: str) -> dict[str, Any]:
    if isinstance(spec.get("model"), tuple):
        spec = spec | {"model": snapshot(spec["model"])}
    return {"model": snapshot(target), "load_format": load_format, "max_model_len": 4096,
            "gpu_memory_utilization": 0.85, "max_num_seqs": 16, "enable_prefix_caching": False,
            "speculative_config": spec, "compilation_config": {"max_cudagraph_capture_size": 128},
            "disable_log_stats": False}  # get_metrics needs stat logging


def snapshot(repo: tuple[str, str]) -> str:
    name, revision = repo
    return f"{os.environ['HF_HUB_CACHE']}/models--{name.replace('/', '--')}/snapshots/{revision}"


def spec_decode_counters(llm) -> dict[str, float]:
    metrics = llm.get_metrics()
    return {m.name: m.value for m in metrics if m.name.startswith("vllm:spec_decode") and hasattr(m, "value")}


def speculator_state(worker) -> dict[str, Any]:
    speculator = worker.model_runner.speculator
    estimator = speculator.acceptance_estimator
    head = getattr(speculator.model.model, "confidence_head", None)
    return {"speculator": type(speculator).__name__,
            "use_confidence_head": getattr(speculator, "use_confidence_head", None),
            "confidence_head_built": head is not None,
            "estimator": None if estimator is None else estimator_state(estimator)}


def estimator_state(estimator) -> dict[str, Any]:
    # predict() writes the features, so all-zero features mean it never ran.
    return {"refits": estimator._refits, "accumulated": float(estimator.counts.sum()),
            "features_nonzero": int((estimator.features != 0).sum())}


if __name__ == "__main__":
    main()
