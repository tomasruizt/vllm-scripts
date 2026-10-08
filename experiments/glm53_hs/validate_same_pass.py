"""Validate extraction against observations of each request's own forward pass."""

import json
import os
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from same_pass_oracle import assert_exact, check_saved_output


def main():
    output = Path(os.environ["HS_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    oracle_dir = output / "oracle"
    assert not (oracle_dir / "manifest.jsonl").exists(), "Use a fresh output directory"
    small = os.getenv("ORACLE_SMALL", "0") == "1"
    layer_ids = [3, 0, 1] if small else [43, 5, 22]
    llm = LLM(**engine_options(output, small, layer_ids))
    print(
        llm.collective_rpc(
            "install_same_pass_oracle", args=(str(oracle_dir), layer_ids)
        ),
        flush=True,
    )
    if small:
        lengths = [7, 127, 128, 129, 255, 256, 257, 1201]
        prompts = [
            {"prompt_token_ids": [(j + i * 31) % 255 + 1 for j in range(n)]}
            for i, n in enumerate(lengths)
        ]
    else:
        lengths = [8, 511, 512, 513, 1151, 1152, 1153, 1921]
        tokenizer = llm.get_tokenizer()
        prompts = []
        for i, length in enumerate(lengths):
            tokens = tokenizer.encode(
                f"Record {i}: A gardener records rainfall and soil moisture each morning. "
                * 300
            )
            assert len(tokens) >= length
            prompts.append({"prompt_token_ids": tokens[:length]})
    cases = [
        ("serial", prompts[:1]),
        ("chunked", prompts[-1:]),
        ("mixed", prompts),
        ("reuse_reordered", list(reversed(prompts))),
        ("identical_prompts", [prompts[2], prompts[2]]),
    ]
    results = []
    negative_controls = False
    for label, batch in cases:
        outputs = llm.generate(
            batch, SamplingParams(temperature=0, max_tokens=1), use_tqdm=False
        )
        assert len(outputs) == len(batch)
        for generated in outputs:
            row, reference = check_saved_output(generated, oracle_dir, layer_ids)
            results.append({"case": label, **row})
            if not negative_controls:
                detected = check_negative_controls(reference)
                assert "corruption" in detected
                if not small:
                    assert len(detected) == 3
                negative_controls = True
    assert any(row["chunks"] > 1 for row in results)
    assert negative_controls
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"PASS: {len(results)} requests match their own forward pass exactly; "
        "layer semantics and negative controls passed",
        flush=True,
    )


def engine_options(output, small, layer_ids):
    options = dict(
        worker_extension_cls="same_pass_oracle.OracleWorkerExtension",
        model=os.environ.get("MODEL_DIR"),
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
            "kv_connector_extra_config": {"shared_storage_path": str(output / "saved")},
        },
    )
    if small:
        from vllm.transformers_utils.configs.glm5_next import Glm5NextTextConfig

        config_dir = output / "model"
        Glm5NextTextConfig(
            architectures=["Glm5NextForCausalLM"],
            vocab_size=256,
            pad_token_id=0,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=16,
            q_lora_rank=128,
            linear_num_heads=4,
            n_routed_experts=None,
            first_k_dense_replace=4,
            layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
            index_head_dim=128,
            index_n_heads=16,
            index_topk=128,
            max_position_embeddings=2048,
        ).save_pretrained(config_dir)
        options.update(
            model=str(config_dir),
            tensor_parallel_size=1,
            skip_tokenizer_init=True,
            load_format="dummy",
            block_size=128,
            max_model_len=1536,
            max_num_batched_tokens=128,
            kv_cache_memory_bytes=64 * 1024**2,
        )
        options.pop("limit_mm_per_prompt")
    return options


def check_negative_controls(reference):
    corrupted = reference.clone()
    corrupted[0, 0, 0] += 1
    detected = []
    for name, changed in [
        ("corruption", corrupted),
        ("layer swap", reference.flip(1)),
        ("token shift", reference.roll(1, 0)),
    ]:
        if torch.equal(changed, reference):
            print(
                f"Uninformative negative control on dummy weights: {name}", flush=True
            )
            continue
        try:
            assert_exact(changed, reference, name)
        except AssertionError:
            detected.append(name)
            continue
        raise AssertionError(f"Oracle did not detect {name}")
    return detected


if __name__ == "__main__":
    main()
