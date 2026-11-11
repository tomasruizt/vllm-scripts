"""Print the automatic CTAs per row of the hierarchical CUDA filter for each batch size.

Usage: python hier_splits.py [VOCAB]
"""

import sys

from fastpath_bench import BATCHES

from hier_cuda import splits

vocab = int(sys.argv[1]) if len(sys.argv) > 1 else 248320
print({rows: splits(rows, vocab) for rows in BATCHES})
