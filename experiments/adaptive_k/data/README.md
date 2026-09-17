# Benchmark prompt provenance

`qualitative.jsonl` is derived from NVIDIA's [SPEED-Bench qualitative test split](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/main/qualitative), specifically its published `test-00000-of-00001.parquet` file. Each JSONL row contains the available prompt as a chat `messages` array plus the original category, source URL, and source ID.

The published split has 880 rows. Some prompt text is replaced by `FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH`; those rows were excluded. The resulting local file has 386 usable rows. It is therefore a subset of the qualitative split, not the complete benchmark used in the vLLM adaptive verification blog. An attempt to fetch the full prompts through the official preparation path stopped at an external Humanity's Last Exam dataset access requirement.

All experiment arms use this same local file. The benchmark uses the `speed_bench` loader, temperature 1.0, top-p 0.95, output cap 2048, and seed 0. The source dataset is under NVIDIA's evaluation dataset license; consult the [dataset card](https://huggingface.co/datasets/nvidia/SPEED-Bench) before redistribution.
