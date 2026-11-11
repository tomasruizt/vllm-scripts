-- Fixed-timing runs (bench_fixed/benchmark_topk_topp.py) to one long-format parquet.
-- Usage: duckdb < fixed_timing.sql (reads results/fixed-timing/GPU/results/{report,glm}.jsonl)
create or replace table raw as
select regexp_extract(filename, 'fixed-timing/([^/]+)/', 1) as gpu,
       regexp_extract(filename, '/(report|glm)\.jsonl', 1) as suite, *
from read_json('results/fixed-timing/*/results/*.jsonl', filename = true, union_by_name = true)
where filename not like '%/b200-old/%' and filename not like '%clone.jsonl';

create or replace table long as
select gpu, suite, Scenario as scenario, Batch as batch, Vocab as vocab, 'triton' as impl,
       "triton (ms)" * 1000 as us, "triton check" as exact, "triton max TV" as max_tv
from raw
union all
select gpu, suite, Scenario, Batch, Vocab, 'new', "final (ms)" * 1000, "final check", "final max TV"
from raw;

copy (from long order by all) to 'results/fixed-timing/results.parquet' (format parquet);
select gpu, suite, impl, count(*) as n, count_if(exact) as exact from long group by all order by all;
