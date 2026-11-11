-- Per-GPU summary of the fixed-timing runs, and the move vs the old timing (call + clone minus clone).
-- Usage: duckdb < fixed_timing_report.sql (after fixed_timing.sql)
create or replace macro peak_tbs(g) as case g when 'h100' then 3.35 when 'a100' then 2.04 when 'rtx' then 1.79 else 8.0 end;
-- SOL: one read and one write of every row the kernel must touch (no-op rows of *_partial can be skipped).
create or replace macro sol_us(g, scenario, batch, vocab) as
  2 * (case when scenario in ('topk_partial', 'topp_partial') then batch // 2 else batch end) * vocab * 4 / (peak_tbs(g) * 1e6);

create or replace table f as pivot (from 'results/fixed-timing/results.parquet') on impl using first(us) group by gpu, suite, scenario, batch, vocab;

select gpu, count_if(new <= 0.9 * triton) as faster10, count_if(new >= 1.1 * triton) as slower10,
       round(100 * (exp(avg(ln(new / triton))) - 1), 0) as geomean_pct,
       round(min(new - triton), 1) as best_delta_us, round(max(new - triton), 1) as worst_delta_us
from f where suite = 'report' group by gpu order by gpu;

select gpu, round(100 * min(new / triton - 1), 0) as glm_best_pct, round(100 * max(new / triton - 1), 0) as glm_worst_pct,
       round(min(new - triton), 1) as glm_best_us, round(max(new - triton), 1) as glm_worst_us
from f where suite = 'glm' group by gpu order by gpu;

select gpu, round(max(100 * sol_us(gpu, scenario, batch, vocab) / new), 0) as max_new_sol,
       round(max(100 * sol_us(gpu, scenario, batch, vocab) / triton), 0) as max_triton_sol,
       round(max(case when scenario like 'real%' then 100 * sol_us(gpu, scenario, batch, vocab) / new end), 0) as max_new_sol_real
from f group by gpu order by gpu;

select gpu, impl, scenario, batch, vocab from 'results/fixed-timing/results.parquet' where not exact and impl = 'new';

-- Old timing: nv-after-rocm (us includes the clone, New kernel = cand) and the MI355X one-pass run.
create or replace table old as
select gpu, suite, scenario, batch, vocab, case impl when 'cand' then 'new' else impl end as impl, us - clone_us as us
from 'results/nv-after-rocm/results.parquet' where suite in ('report', 'glm') and impl in ('triton', 'cand')
  and gpu in ('b300', 'h100', 'a100', 'rtx')
union all
select 'mi355x', 'report', Scenario, Batch, Vocab, 'triton', ("triton (ms)" - "clone-only (ms)") * 1000 from 'results/amd-port/onepass6.jsonl'
union all
select 'mi355x', 'report', Scenario, Batch, Vocab, 'new', ("triton-hier (ms)" - "clone-only (ms)") * 1000 from 'results/amd-port/onepass6.jsonl';

select n.gpu, n.impl, n.batch, round(100 * (exp(avg(ln(n.us / o.us))) - 1), 0) as moved_pct, round(median(n.us - o.us), 1) as median_moved_us
from 'results/fixed-timing/results.parquet' n join old o using (gpu, suite, scenario, batch, vocab, impl)
where n.suite = 'report' and n.batch in (8, 64, 2048) group by all order by all;
