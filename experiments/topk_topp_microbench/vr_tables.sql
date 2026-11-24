-- Vera Rubin A/B (vr_run.sh): long-format parquet, fastest of the passes per case and impl.
-- Usage, in the fetched run folder: cd results/RUN_ID && duckdb < ../../vr_tables.sql
create or replace table raw as
select regexp_extract(filename, '/(report|glm)_r(\d+)', 1) as suite, regexp_extract(filename, '_r(\d+)', 1)::int as pass,
       Scenario as scenario, Batch as batch, Vocab as vocab,
       "triton (ms)" * 1000 as triton_us, "new (ms)" * 1000 as new_us, "triton check" as triton_exact, "new check" as new_exact
from read_json('results/*_r*.jsonl', filename = true);
copy (
  select 'rubin' as gpu, suite, scenario, batch, vocab, 'triton' as impl, min(triton_us) as us, bool_and(triton_exact) as exact from raw group by all
  union all
  select 'rubin', suite, scenario, batch, vocab, 'new', min(new_us), bool_and(new_exact) from raw group by all
) to 'results.parquet';
