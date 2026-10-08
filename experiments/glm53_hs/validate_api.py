"""Use Speculators' real HTTP client and file checks against a running server."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
from safetensors.torch import load_file
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    generate_hidden_states_async,
    wait_for_lock_async,
)
from transformers import AutoConfig, AutoTokenizer

from same_pass_oracle import check_saved_output
from validate_same_pass import check_negative_controls


async def main():
    output = Path(os.environ["HS_OUTPUT_DIR"])
    endpoint = os.environ.get("HS_ENDPOINT", "http://127.0.0.1:8150")
    config = AutoConfig.from_pretrained(os.environ["MODEL_DIR"])
    config = getattr(config, "text_config", config)
    layers = [5, 22, 43, config.num_hidden_layers]
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"])
    prompts = [tokenizer.encode(f"Record {i}: A gardener records rainfall and soil moisture each morning. " * 300)[:n]
               for i, n in enumerate([8, 511, 512, 513, 1151, 1152, 1153, 1921])]
    cases = [("serial", prompts[:1]), ("chunked", prompts[-1:]),
             ("mixed", prompts), ("reuse_reordered", list(reversed(prompts))),
             ("identical", [prompts[2], prompts[2]])]
    results = []
    async with httpx.AsyncClient(timeout=60) as admin:
        await rpc(admin, endpoint, "install_api_oracle", [str(output / "oracle")])
        async with openai.AsyncOpenAI(base_url=endpoint + "/v1", api_key="unused", max_retries=0) as client:
            models = await client.models.list()
            model = models.data[0].id
            for phase in ["client_only", "same_replay_reference"]:
                before = await rpc(admin, endpoint, "api_oracle_status", ["true" if phase == "same_replay_reference" else "false"])
                for label, batch in cases:
                    rows = await asyncio.gather(*[
                        check_request(client, model, tokens, output, layers, phase, label)
                        for tokens in batch
                    ])
                    results.extend(rows)
                    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                after = await rpc(admin, endpoint, "api_oracle_status", ["false"])
                delta = after["target_replays"] - before["target_replays"]
                (output / f"{phase}-graphs.json").write_text(json.dumps({"before": before, "after": after, "target_replays": delta}, indent=2) + "\n")
                assert delta > 0, f"No target CUDA graph replay observed in {phase}"
                print(f"PASS {phase}: 20 HTTP requests; {delta} target graph replays", flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


async def check_request(client, model, tokens, output, layers, phase, label):
    path = await generate_hidden_states_async(client, model, {"input_ids": tokens}, timeout=300, max_retries=0)
    if Path(path + ".lock").exists():
        await wait_for_lock_async(path + ".lock", timeout=60)
    data = load_file(path)
    check_hidden_states(data, tokens)
    assert data["hidden_states"].shape[1] == len(layers)
    row = {"phase": phase, "case": label, "tokens": len(tokens), "path": path}
    if phase == "same_replay_reference":
        generated = SimpleNamespace(kv_transfer_params={"hidden_states_path": path}, prompt_token_ids=tokens)
        checked, reference = check_saved_output(generated, output / "oracle", layers, actual=data)
        row.update(checked)
        fragments = [json.loads(line) for line in (output / "oracle/manifest.jsonl").read_text().splitlines()]
        fragments = [entry for entry in fragments if entry["request_id"] == row["request_id"]]
        row["all_chunks_replayed_graphs"] = all(entry["target_replays"] > 0 for entry in fragments)
        assert row["all_chunks_replayed_graphs"], "A reference chunk did not replay CUDA graphs"
        if label == "serial":
            row["negative_controls_detected"] = check_negative_controls(reference)
            assert len(row["negative_controls_detected"]) == 3
    return row


async def rpc(client, endpoint, method, args):
    response = await client.post(endpoint + "/collective_rpc", json={"method": method, "args": args})
    response.raise_for_status()
    return next(item for item in response.json()["results"] if item["observing"])


if __name__ == "__main__":
    asyncio.run(main())
