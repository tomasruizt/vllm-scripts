-- All implementations on the same 140 configs: one pass of results/hier-cuda/<gpu>/all-<gpu>.jsonl
-- (triton, cuda, triton-core = New Triton, hier = CUDA hierarchical, fi-opt) plus CuTe from its
-- 3-pass run (median). Kernel time = call time minus the input clone (results/fiopt6).
-- Usage: duckdb < all_impls.sql
create or replace temp table clone as
select gpu, Batch batch, Vocab vocab, median(1000 * "clone-only (ms)") clone
from read_json_auto('results/fiopt6/*/*.jsonl', union_by_name = true, filename = true)
cross join lateral (select split_part(filename, '/', 3) gpu)
group by all;

create or replace temp table raw as
select split_part(filename, '/', 3) gpu, Scenario scenario, Batch batch, Vocab vocab,
       "hier (ms)" hier, "cuda (ms)" cuda, "triton (ms)" triton, "triton-core (ms)" "new-triton",
       "fi-opt (ms)" fi, "hier check" hier_ok, "cuda check" cuda_ok, "triton check" triton_ok,
       "triton-core check" "new-triton_ok", "fi-opt check" fi_ok
from read_json_auto('results/hier-cuda/*/all-*.jsonl', union_by_name = true, filename = true);

create or replace temp table t as
select gpu, scenario, batch, vocab, impl, 1000 * ms us, exact
from (unpivot raw on (hier, hier_ok) as hier, (cuda, cuda_ok) as cuda, (triton, triton_ok) as triton,
      ("new-triton", "new-triton_ok") as "new-triton", (fi, fi_ok) as fi into name impl value ms, exact)
union all
select gpu, scenario, batch, vocab, 'cute', 1000 * median(time_ms), bool_and(exact)
from 'results/cute-final/cute-final.parquet' where kind = 'pass' and impl = 'cute' group by all;

create or replace temp table k as
select t.*, us - clone us_k from t join clone using (gpu, batch, vocab);
