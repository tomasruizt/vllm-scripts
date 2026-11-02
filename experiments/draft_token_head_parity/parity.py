"""Check that the draft-token-head refactor keeps DFlash, XPress and DSpark drafting unchanged."""
import hashlib
import json
import os
import sys
from typing import Any

QWEN3 = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
DFLASH = ("z-lab/Qwen3-8B-DFlash-b16", "9b41424b7109f9c5413454f481b09a82b85333f4")
XPRESS = ("UIUC-SSAIL/Qwen3-8B-XPress-b16", "39ab6e7ffbd0eb95b98ae799203434c98c734673")
DSV4 = ("inference-optimization/DeepSeek-V4-Flash-DSpark-2.1B-A1.9B-small", "bbddd703cacc556881da865ecb67663454e943ab")
AV = {"enable_adaptive_verification": True}
DSPARK = {"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic"} | AV
# name -> (target, draft, speculative config extras, load format, sampling temperature)
VARIANTS: dict[str, tuple[tuple[str, str], tuple[str, str] | None, dict[str, Any], str, float]] = {
    "dflash_greedy": (QWEN3, DFLASH, {}, "auto", 0.0),
    "dflash_greedy_av": (QWEN3, DFLASH, AV, "auto", 0.0),
    "dflash_prob": (QWEN3, DFLASH, {"draft_sample_method": "probabilistic"}, "auto", 1.0),
    "xpress_greedy": (QWEN3, XPRESS, {}, "auto", 0.0),
    "xpress_greedy_av": (QWEN3, XPRESS, AV, "auto", 0.0),
    "dspark_av": (DSV4, None, DSPARK, "dummy", 1.0),
    "dspark_online": (DSV4, None, DSPARK | {"confidence_head": "online_estimator"}, "dummy", 1.0),
}


def main() -> None:
    variant = sys.argv[1]
    target, draft, extras, load_format, temperature = VARIANTS[variant]
    spec = ({"method": "dflash", "model": snapshot(draft), "num_speculative_tokens": 15} if draft else {}) | extras
    result = {"variant": variant, "commit": os.environ.get("VLLM_COMMIT"), "spec": spec}
    print("RESULT " + json.dumps(result | run(target, spec, load_format, temperature)), flush=True)


def run(target: tuple[str, str], spec: dict[str, Any], load_format: str, temperature: float) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=snapshot(target), load_format=load_format, max_model_len=4096, gpu_memory_utilization=0.85,
              max_num_seqs=16, enable_prefix_caching=False, speculative_config=spec, disable_log_stats=False,
              compilation_config={"max_cudagraph_capture_size": 128})
    prompts = [f"Write a story about the number {i}." for i in range(16)]
    params = [SamplingParams(max_tokens=256, ignore_eos=True, temperature=temperature, seed=i) for i in range(16)]
    token_ids = [list(output.outputs[0].token_ids) for output in llm.generate(prompts, params)]
    digest = hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()[:16]
    return {"num_output_tokens": sum(map(len, token_ids)), "tokens_sha256": digest,
            "spec_decode": spec_decode_counters(llm), "workers": llm.collective_rpc(speculator_state)}


def snapshot(repo: tuple[str, str]) -> str:
    name, revision = repo
    return f"{os.environ['HF_HUB_CACHE']}/models--{name.replace('/', '--')}/snapshots/{revision}"


def spec_decode_counters(llm) -> dict[str, float]:
    metrics = llm.get_metrics()
    return {m.name: m.value for m in metrics if m.name.startswith("vllm:spec_decode") and hasattr(m, "value")}


def speculator_state(worker) -> dict[str, Any]:
    speculator = worker.model_runner.speculator
    head = getattr(speculator, "draft_token_head", None)
    estimator = speculator.acceptance_estimator
    return {"speculator": type(speculator).__name__, "draft_token_head": head and type(head).__name__,
            "estimator": None if estimator is None else {"refits": estimator._refits,
                                                         "features_nonzero": int((estimator.features != 0).sum())}}


if __name__ == "__main__":
    main()
