# Spec decode count invariant check

Checks `drafted >= verified >= accepted` (vllm-project/vllm#64156) against live servers: at every step of the `detailed` per-request metrics, on the Prometheus counter deltas, and that the `verified` per-position counters sum to the total. With adaptive verification it also checks `verified < drafted`.

- [check.py](check.py): sends one JSON-schema and one free-text request (temperature 0) to a running server and asserts the invariants.
- [run_one.sh](run_one.sh): starts `vllm serve` for one config, waits for health, runs `check.py`.
- [run_all.sh](run_all.sh): runs all configs in parallel with `chg run` (one GPU each): `ngram_gpu`, EAGLE-3 and DSpark with and without AV, sync and async, plus DSpark + AV on the Rust frontend.

```bash
source <vllm-tree>/.venv/bin/activate
export HF_HOME=/data/tomasruizt/huggingface
bash run_all.sh  # results in results/<config>/check.json
```

The Rust frontend config uses the `vllm-rs` binary shipped in `vllm/`. With a precompiled wheel that binary is built from main, so build it from the tree under test (`cargo build --release --bin vllm-rs` in `rust/`) and copy it to `vllm/vllm-rs` first.
