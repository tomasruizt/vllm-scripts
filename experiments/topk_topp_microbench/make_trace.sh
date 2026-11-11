#!/usr/bin/env bash
# Instrument csrc/topk_topp_filter.cu with clock64 phase traces of CTA 0 and a
# TT_SPLITS override. Usage: make_trace.sh SRC OUT
set -euo pipefail
SRC=$1 OUT=$2
before() { sed -i.bak "/$1/i\\
$2
" "$OUT"; }
after() { sed -i.bak "/$1/a\\
$2
" "$OUT"; }
cp "$SRC" "$OUT"
after '^namespace topk_topp_filter {' '__shared__ unsigned long long trace_ts[64];\
__shared__ int trace_tag[64];\
__shared__ unsigned trace_n;\
#define TR(tag) do { if (threadIdx.x == 0 \&\& blockIdx.x == 0) { unsigned i = trace_n++; if (i < 64) { trace_ts[i] = clock64(); trace_tag[i] = tag; } } } while (0)\
__device__ void trace_dump() {\
  if (threadIdx.x || blockIdx.x) return;\
  unsigned n = min(trace_n, 64u);\
  for (unsigned i = 1; i < n; ++i) printf("TRACE %d->%d %llu\\n", trace_tag[i-1], trace_tag[i], trace_ts[i] - trace_ts[i-1]);\
  trace_n = 0;\
}'
after '^#include <mutex>' '#include <cstdlib>'
sed -i.bak 's/^  const int splits = tf::choose_splits(pl, int(batch), int(vocab));/  const char* force = getenv("TT_SPLITS");\
  const int splits = force ? atoi(force) : tf::choose_splits(pl, int(batch), int(vocab));/' "$OUT"
# process_row phases
after '^    tk = sample_threshold<CL>(s, v, a);' '    TR(1);'
after '^  Stats st = stream_row<CL>(s, store, ring, x, begin, end, vec, a);' '  TR(2);'
before '^  bool restarted =' '  TR(3);'
before '^  Cut set =$' '  TR(4);'
before '^      unsigned lower =$' '      TR(5);'
before '^      if (lower != NO_CANDIDATES) {' '      TR(6);'
before '^      if (cand_mass < target) {' '      TR(7);'
before '^  \/\/ Large kept sets' '  TR(9);'
after '^      tk = compact(s, st, v, base, tk, lost' '      TR(40);'
# radix rounds
before '^    visit(\[&\](unsigned key, unsigned value) {' '    TR(31);'
before '^    if (CL) push_bins<W>(s, nb);' '    TR(32);'
after '^    if (CL) push_bins<W>(s, nb);' '    TR(33);'
before '^    found = s.found;' '    TR(34);'
# kernels
before '^    process_row<false>(s, store, ring' '    TR(0);'
before '^    if (row_params(a, row, use_k, kk, use_p, p))$' '    TR(0);'
after '^    row = next;' '    TR(20); trace_dump();'
before '^    \/\/ Peers finish reading this' '    TR(20); trace_dump();'
rm -f "$OUT.bak"
grep -c "TR(" "$OUT"
# streaming rounds (tag 40 = round end)
sed -i.bak 's/^      TR(40);$//' "$OUT"
before '^    if (__syncthreads_or(lost != 0))$' '    TR(42);'
sed -i.bak 's/^      tk = compact(s, st, v, base, tk, lost, a.keep_rank, a.min_needed, status);$/      tk = compact(s, st, v, base, tk, lost, a.keep_rank, a.min_needed, status);\
    TR(40);/' "$OUT"
before '^    unsigned lost =$' '    TR(41);'
rm -f "$OUT.bak"
after '^__device__ void init_ring(Shared& s, int depth) {' '  trace_n = 0;'
rm -f "$OUT.bak"
after '^        "shared memory size");' '  p.max_smem -= 1024;  \/\/ static trace buffers'
rm -f "$OUT.bak"
