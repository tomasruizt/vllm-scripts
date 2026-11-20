-- The PR head's kernel vs the kernel revision of the PR description (final) and main's Triton,
-- from one nv_after_rocm.py run with --impls triton final head.
-- Usage: NAME=h100 GPU=h100 duckdb < head_regression.sql  (results/pr66792-head/$NAME-{report,gate})
create or replace macro us(x) as 1000 * x;
create or replace table r as
  select 'report' as suite, * from read_json_auto('results/pr66792-head/' || getenv('NAME') || '-report/report.jsonl')
  union all by name
  select 'glm' as suite, * from read_json_auto('results/pr66792-head/' || getenv('NAME') || '-report/glm.jsonl');
create or replace table t as
  select suite, Scenario as scenario, Batch as batch, Vocab as vocab, us("triton (ms)") as triton, us("final (ms)") as final,
         us("head (ms)") as head, "head check" as head_exact, "final check" as final_exact from r;

.print '== head vs the description''s kernel (same process)'
select suite, count(*) as cases, count_if(not head_exact) as head_inexact,
       round(100 * (exp(avg(ln(head / final))) - 1), 1) as geomean_pct,
       count_if(head >= 1.10 * final) as slower10, count_if(head >= 1.03 * final) as slower3, count_if(head <= 0.97 * final) as faster3,
       round(100 * max(head / final - 1), 1) as worst_pct, round(max(head - final), 1) as worst_delta_us
from t group by suite order by suite desc;

.print '== cases where head is >= 3% slower than the description''s kernel'
select suite, scenario, batch, vocab, round(final, 1) as final_us, round(head, 1) as head_us,
       round(100 * (head / final - 1), 1) as pct, round(head - final, 1) as delta_us
from t where head >= 1.03 * final order by head / final desc limit 15;

.print '== claims table: this run (head and final) vs the stored run of the description'
with now as (
  select k.impl, count_if(k.x <= 0.9 * triton) as faster10, count_if(k.x >= 1.1 * triton) as slower10,
         round(100 * (exp(avg(ln(k.x / triton))) - 1), 0) as geomean_pct,
         round(min(k.x - triton), 1) as best_delta_us, round(max(k.x - triton), 1) as worst_delta_us
  from t, lateral (values ('head now', head), ('final now', final)) k(impl, x) where suite = 'report' group by k.impl),
old as (
  select 'description' as impl, count_if(new <= 0.9 * triton) as faster10, count_if(new >= 1.1 * triton) as slower10,
         round(100 * (exp(avg(ln(new / triton))) - 1), 0) as geomean_pct,
         round(min(new - triton), 1) as best_delta_us, round(max(new - triton), 1) as worst_delta_us
  from (pivot (from 'results/fixed-timing/results.parquet' where gpu = getenv('GPU') and suite = 'report')
        on impl using first(us) group by scenario, batch, vocab))
select * from old union all select * from now order by impl;

.print '== real rows: head now vs the description''s stored numbers'
with old as (pivot (from 'results/fixed-timing/results.parquet' where gpu = getenv('GPU')) on impl using first(us) group by suite, scenario, batch, vocab)
select t.suite, t.scenario, t.batch, round(old.triton, 1) as desc_triton, round(old.new, 1) as desc_new,
       round(t.triton, 1) as triton_now, round(t.final, 1) as final_now, round(t.head, 1) as head_now,
       round(100 * (t.head / t.final - 1), 1) as head_vs_final_pct
from t join old on (t.suite = old.suite and t.scenario = old.scenario and t.batch = old.batch and t.vocab = old.vocab)
where t.scenario like 'real%' order by t.suite desc, t.scenario, t.batch;
