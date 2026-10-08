import os
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    load_hidden_states,
)


def main():
    layer_ids = [5, 22, 43]
    output_dir = Path(os.environ["HS_OUTPUT_DIR"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=os.environ["MODEL_DIR"],
        tensor_parallel_size=int(os.getenv("TP_SIZE", "4")),
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(output_dir)},
        },
    )
    tokenizer = llm.get_tokenizer()
    texts = [
        "Explain why the sky appears blue.",
        "A gardener records rainfall and soil moisture each morning. " * 80,
        "A compiler translates source code and optimizes the resulting program. " * 160,
    ]
    prompts = [{"prompt_token_ids": tokenizer.encode(text)} for text in texts]
    assert all(0 < len(p["prompt_token_ids"]) < 4096 for p in prompts)
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    params = SamplingParams(temperature=0, max_tokens=1)

    references = []
    for prompt in prompts:
        (output,) = llm.generate([prompt], params)
        references.append(read_checked(output, len(layer_ids), hidden_size))

    for repeat in range(2):
        outputs = llm.generate(prompts, params)
        assert len(outputs) == len(references)
        for index, (output, reference) in enumerate(zip(outputs, references)):
            actual = read_checked(output, len(layer_ids), hidden_size)
            delta = (actual.float() - reference.float()).abs()
            print(f"batch={repeat} request={index} shape={tuple(actual.shape)} "
                  f"max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g}",
                  flush=True)
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-2)
    print("PASS: full-checkpoint serial/concurrent hidden-state extraction")


def read_checked(output, num_layers, hidden_size):
    path = output.kv_transfer_params["hidden_states_path"]
    tensors = load_hidden_states(path)
    states = tensors["hidden_states"]
    assert torch.equal(tensors["token_ids"], torch.tensor(output.prompt_token_ids))
    assert states.shape == (len(output.prompt_token_ids), num_layers, hidden_size)
    assert torch.isfinite(states).all()
    assert torch.count_nonzero(states).item() > 0
    return states


if __name__ == "__main__":
    main()
