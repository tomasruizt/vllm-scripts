-- Kernel B vs kernel A from one nv_after_rocm.py run (same process, interleaved), median over the gate's runs.
-- Usage: DIR=results/pr66792-head/b200chunk A=head B=chunk duckdb < ab_kernels.sql
create or replace macro us(x) as 1000 * x;
create or replace table raw as
  select regexp_extract(filename, '-(gate|report)/', 1) as suite, *
  from read_json_auto(getenv('DIR') || '-*/*.jsonl', union_by_name = true, filename = true);
create or replace table t as
  select suite, Scenario as scenario, Batch as batch, Vocab as vocab,
         median(us(columns(getenv('A') || ' \(ms\)'))) as a, median(us(columns(getenv('B') || ' \(ms\)'))) as b,
         bool_and(columns(getenv('B') || ' check')) as b_exact
  from raw group by all;

.print '== B vs A'
select suite, count(*) as cases, count_if(not b_exact) as b_inexact, round(100 * (exp(avg(ln(b / a))) - 1), 1) as geomean_pct,
       count_if(b >= 1.10 * a) as slower10, count_if(b >= 1.03 * a) as slower3, count_if(b <= 0.97 * a) as faster3,
       round(100 * max(b / a - 1), 1) as worst_pct, round(max(b - a), 1) as worst_delta_us
from t group by suite order by suite;

.print '== cases where B differs from A by >= 3%'
select suite, scenario, batch, vocab, round(a, 1) as a_us, round(b, 1) as b_us, round(100 * (b / a - 1), 1) as pct, round(b - a, 1) as delta_us
from t where b >= 1.03 * a or b <= 0.97 * a order by b / a desc limit 40;
