#!/usr/bin/env python3
"""Plot TPOT vs context length using binned boxplots."""

import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Load the benchmark results
results_file = "/home/tomasruiz/code/vllm-scripts/slurm/online-throughput/results/openai-32.0qps-concurrency32-Qwen3-4B-20260122-232333.json"

with open(results_file) as f:
    data = json.load(f)

tpots = data["tpots"]
input_lens = data["input_lens"]
sse_contents = data["sse_content"]

# Load tokenizer to count tokens per SSE
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(data["tokenizer_id"])

# Collect all (context_length, tpot) pairs, excluding first SSE
context_lengths = []
tpot_values = []

for request_idx, request_tpots in enumerate(tpots):
    input_len = input_lens[request_idx]
    sse_content = sse_contents[request_idx]

    # Compute cumulative token count for each SSE
    cumulative_tokens = 0
    for idx, (tpot, content) in enumerate(zip(request_tpots, sse_content)):
        # Count tokens in this SSE
        if content:
            num_tokens = len(tokenizer(content, add_special_tokens=False).input_ids)
        else:
            num_tokens = 0

        cumulative_tokens += num_tokens

        if idx == 0:  # Skip first SSE (TPOT=0 by design)
            continue

        # Context length = input tokens + output tokens generated so far
        context_lengths.append(input_len + cumulative_tokens)
        tpot_values.append(tpot * 1000)  # Convert to ms

context_lengths = np.array(context_lengths)
tpot_values = np.array(tpot_values)

print(f"Total data points: {len(context_lengths)}")
print(f"Context length range: {context_lengths.min()} - {context_lengths.max()}")

# Create bins for context length
bin_size = 500  # tokens per bin
min_ctx = int(np.floor(context_lengths.min() / bin_size) * bin_size)
max_ctx = int(np.ceil(context_lengths.max() / bin_size) * bin_size)
bin_edges = np.arange(min_ctx, max_ctx + bin_size, bin_size)

# Assign each point to a bin
bin_indices = np.digitize(context_lengths, bin_edges) - 1

# Group TPOT values by bin
binned_data = []
bin_labels = []
bin_counts = []

for i in range(len(bin_edges) - 1):
    mask = bin_indices == i
    bin_tpots = tpot_values[mask]
    if len(bin_tpots) >= 10:  # Only include bins with enough samples
        binned_data.append(bin_tpots)
        bin_labels.append(f"{bin_edges[i]}-{bin_edges[i + 1]}")
        bin_counts.append(len(bin_tpots))

print(f"Number of bins with >=10 samples: {len(binned_data)}")

# Create the boxplot
fig, ax = plt.subplots(figsize=(14, 6))

bp = ax.boxplot(
    binned_data,
    patch_artist=True,
    showfliers=False,
    whis=(1, 99),  # Set whiskers to p1 and p99
)

# Color the boxes
for patch in bp["boxes"]:
    patch.set_facecolor("lightblue")
    patch.set_alpha(0.7)

# Set x-axis labels
ax.set_xticks(range(1, len(bin_labels) + 1))
ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)

ax.set_xlabel("Context Length (tokens)", fontsize=12)
ax.set_ylabel("TPOT (ms)", fontsize=12)
ax.set_title(
    "Time Per Output Token vs Context Length (Binned)\n(Qwen3-4B with Speculative Decoding, k=4)",
    fontsize=14,
)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 50)  # Clip y-axis to focus on main distribution

# Add sample counts as text above each box
for i, count in enumerate(bin_counts):
    ax.text(
        i + 1,
        ax.get_ylim()[1] * 0.95,
        f"n={count}",
        ha="center",
        va="top",
        fontsize=7,
        color="gray",
    )

plt.tight_layout()
plt.savefig(
    "/home/tomasruiz/code/vllm-scripts/slurm/online-throughput/results/tpot_vs_context_boxplot.png",
    dpi=150,
)
print(f"Plot saved to results/tpot_vs_context_boxplot.png")

# Print statistics per bin
print(f"\nPer-bin statistics:")
print(f"{'Bin':<12} {'Count':>6} {'Median':>10} {'P99':>10}")
print("-" * 42)
for i, (label, data_bin) in enumerate(zip(bin_labels, binned_data)):
    median = np.median(data_bin)
    p99 = np.percentile(data_bin, 99)
    print(f"{label:<12} {len(data_bin):>6} {median:>10.2f} {p99:>10.2f}")
