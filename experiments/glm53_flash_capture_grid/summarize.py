"""Collect the GLM pilot using the same validity gates as the Qwen experiment."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen38_capture_grid"))
from summarize import load_cell


def main() -> None:
    root = Path(sys.argv[1])
    rows = [glm_cell(path, root) for path in sorted(root.glob("*/artifacts/**/profile_export_aiperf.json"))]
    expected = {(variant, concurrency, repeat) for variant in ("Default", "Up to 4K") for concurrency in (64, 128) for repeat in (1, 2, 3)}
    assert len(rows) == 12 and {(row["variant"], row["concurrency"], row["repeat"]) for row in rows} == expected
    (root / "metrics.json").write_text(json.dumps(rows) + "\n")
    for row in rows:
        print({key: row[key] for key in ("variant", "concurrency", "repeat", "valid", "failed_gates", "throughput", "mean_tpot_ms", "window_s")})


def glm_cell(path: Path, root: Path) -> dict:
    row = load_cell(path, root)
    directory = root / path.relative_to(root).parts[0]
    info = json.loads((directory / "server-info.json").read_text())
    config = info["vllm_config"]
    sizes = config["compilation_config"]["cudagraph_capture_sizes"]
    row.update(model="zai-org/GLM-5.3-Flash", model_revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
               weight_dtype="fp8", prefix_caching=True, expert_parallel=config["parallel_config"]["enable_expert_parallel"],
               num_speculative_tokens=5, embedding_cpu_offload=False,
               max_model_len=config["model_config"]["max_model_len"],
               max_num_seqs=config["scheduler_config"]["max_num_seqs"],
               max_num_batched_tokens=config["scheduler_config"]["max_num_batched_tokens"],
               capture_cap=max(sizes), capture_sizes=sizes, server_info_json=json.dumps(info))
    return row


if __name__ == "__main__":
    main()
