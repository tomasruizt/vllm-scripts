-- Fixed-timing tables after the fallback queue (#66792 at b2e00c68cc): Triton from the fixed-timing runs,
-- the new kernel from the first pass of the queue A/B runs (fix6 = the pushed kernel), like the one
-- Triton pass. A100: a fresh run of both on another node (results/fb-vet/a100b, fastest of 3 passes),
-- since the earlier A100 node was noisy.
-- Usage: duckdb < fixed_timing_queue.sql   (writes results/fixed-timing-queue/results.parquet)
create or replace table ab as
select regexp_extract(filename, 'fb-vet/([^/]+)/', 1) as gpu, case when Source = 'glm' then 'glm' else 'report' end as suite,
       Scenario as scenario, Batch as batch, Vocab as vocab,
       min("fix6 (ms)") * 1000 as new_us, min("triton (ms)") * 1000 as triton_us, bool_and("fix6 check") as exact
from read_json('results/fb-vet/*/report/report_r*.jsonl', filename = true, union_by_name = true)
where filename not like '%/a100/%' and (filename like '%/a100b/%' or filename like '%report_r0.jsonl') group by all;

create or replace table amd as
select 'mi355x' as gpu, case when source = 'glm' then 'glm' else 'report' end as suite,
       regexp_extract(case_name, '^(.*)_b[0-9]+_v[0-9]+k$', 1) as scenario, batch, vocab, min(us) as new_us, bool_and(exact) as exact
from 'results/fallback-balance/results.parquet' where gpu = 'mi355x' and suite = 'f6-report' and impl = 'fix6' group by all;

create or replace table long as
select gpu, suite, scenario, batch, vocab, 'triton' as impl, us, exact from 'results/fixed-timing/results.parquet'
where impl = 'triton' and gpu not in ('a100')
union all
select gpu, suite, scenario, batch, vocab, 'triton', triton_us, false from ab where gpu = 'a100b'
union all
select gpu, suite, scenario, batch, vocab, 'new', new_us, exact from ab
union all
select gpu, suite, scenario, batch, vocab, 'new', new_us, exact from amd;
update long set gpu = 'a100' where gpu = 'a100b';

copy (from long order by all) to 'results/fixed-timing-queue/results.parquet' (format parquet);
select gpu, suite, impl, count(*) as n, count_if(exact) as exact from long group by all order by all;
