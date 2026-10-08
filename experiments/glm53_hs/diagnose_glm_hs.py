import json
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
    comparisons = []
    for prompt in prompts:
        (output,) = llm.generate([prompt], params)
        references.append(read_checked(output, len(layer_ids), hidden_size))

    for index, prompt in enumerate(prompts):
        (output,) = llm.generate([prompt], params)
        actual = read_checked(output, len(layer_ids), hidden_size)
        comparisons.append(compare("serial_repeat", index, actual, references[index]))

    previous_batch = None
    for repeat in range(2):
        outputs = llm.generate(prompts, params)
        assert len(outputs) == len(references)
        batch_states = []
        for index, (output, reference) in enumerate(zip(outputs, references)):
            actual = read_checked(output, len(layer_ids), hidden_size)
            batch_states.append(actual)
            comparisons.append(compare(f"batch{repeat}_vs_serial", index, actual, reference))
            if previous_batch is not None:
                comparisons.append(compare("batch_repeat", index, actual, previous_batch[index]))
        previous_batch = batch_states

    for index, prompt in enumerate(prompts):
        (output,) = llm.generate([prompt], params)
        actual = read_checked(output, len(layer_ids), hidden_size)
        comparisons.append(compare("serial_after_batch", index, actual, references[index]))

    (output_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    assert all(row["close"] for row in comparisons), "Document tolerance failed; see comparisons.json"
    print("PASS: full-checkpoint extraction and repeatability checks")


def compare(kind, index, actual, reference):
    a, r = actual.float(), reference.float()
    delta = (a - r).abs()
    row = {"kind": kind, "request": index, "shape": list(actual.shape)}
    try:
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-2)
        row["close"] = True
    except AssertionError:
        row["close"] = False
    row["layers"] = []
    for layer in range(actual.shape[1]):
        d, ref, act = delta[:, layer], r[:, layer], a[:, layer]
        row["layers"].append({
            "layer": [5, 22, 43][layer],
            "max_abs": d.max().item(),
            "mean_abs": d.mean().item(),
            "mismatch_fraction": (d > 1e-5 + 1e-2 * ref.abs()).float().mean().item(),
            "relative_l2": (d.norm() / ref.norm()).item(),
            "cosine": torch.nn.functional.cosine_similarity(act.double().flatten(), ref.double().flatten(), dim=0).item(),
        })
    print(json.dumps(row), flush=True)
    return row


def read_checked(output, num_layers, hidden_size):
    path = output.kv_transfer_params["hidden_states_path"]
    tensors = load_hidden_states(path)
    states = tensors["hidden_states"]
    assert torch.equal(tensors["token_ids"], torch.tensor(output.prompt_token_ids))
    assert states.shape == (len(output.prompt_token_ids), num_layers, hidden_size)
    assert torch.isfinite(states).all()
    assert torch.count_nonzero(states).item() > 0
    print(json.dumps({"saved": path, "shape": list(states.shape), "generated_token_ids": output.outputs[0].token_ids}), flush=True)
    return states


if __name__ == "__main__":
    main()
