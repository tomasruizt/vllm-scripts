"""Capture actual GDN kernel execution during generation, separate from benchmarks."""

import argparse
import os
import sys
from pathlib import Path

from run import DRAFT, DRAFT_27B, DRAFT_MOE, TARGET, TARGET_27B, TARGET_MOE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("4B", "27B", "35B-A3B"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    target, draft = {"4B": (TARGET, DRAFT), "27B": (TARGET_27B, DRAFT_27B),
                     "35B-A3B": (TARGET_MOE, DRAFT_MOE)}[args.model]
    llm = LLM(
        model=snapshot_download(target[0], revision=target[1], local_files_only=True),
        dtype="bfloat16", quantization="fp8", tensor_parallel_size=1,
        language_model_only=True, trust_remote_code=True,
        mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="bfloat16",
        max_model_len=32768, max_num_seqs=32, max_num_batched_tokens=2048,
        gpu_memory_utilization=0.92, enable_prefix_caching=False,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]},
        speculative_config={"method": "dflash", "num_speculative_tokens": 7,
                            "model": snapshot_download(draft[0], revision=draft[1], local_files_only=True),
                            "attention_backend": "TRITON_ATTN"},
        profiler_config={"profiler": "torch", "torch_profiler_dir": str(args.output.resolve()),
                         "torch_profiler_with_stack": False},
    )
    prompt = "Solve step by step: A store has 120 apples, sells 35, and receives 48. How many apples remain?"
    params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    llm.generate([prompt], params, use_tqdm=False)
    llm.start_profile()
    try:
        for batch in (1, 32):
            outputs = llm.generate([prompt] * batch, params, use_tqdm=False)
            assert len(outputs) == batch
            assert all(len(output.outputs[0].token_ids) == 64 for output in outputs)
    finally:
        llm.stop_profile()
    print(f"Completed profiled generation for {args.model} at batch sizes 1 and 32.")


if __name__ == "__main__":
    main()
