-- CUDA columns of report/triton-core-vs-cuda-vs-fi-20261109.md from one pass of pr_all.sh BENCH=1 at 3d966d50dc
-- (results/pr-all/bench-<gpu>/pass-0.jsonl). Kernel time = call time minus the input clone
-- (results/fiopt6). Each % is vs. the Triton of the same pass. Usage: duckdb < cuda_report.sql
create or replace temp table clone as
select gpu, Batch batch, Vocab vocab, median(1000 * "clone-only (ms)") clone
from read_json_auto('results/fiopt6/*/*.jsonl', union_by_name = true, filename = true)
cross join lateral (select split_part(filename, '/', 3) gpu)
group by all;

create or replace temp table fi_ok as
select distinct split_part(filename, '/', 3) gpu, Scenario scenario, Batch batch, Vocab vocab
from read_json_auto('results/hier-cuda/*/all-*.jsonl', union_by_name = true, filename = true)
where "fi-opt (ms)" is not null;

create or replace temp table p as
select replace(split_part(filename, '/', 3), 'bench2-', '') gpu, Scenario scenario, Batch batch, Vocab vocab,
       1000 * "triton (ms)" t_ms, 1000 * "cuda (ms)" c_ms, "cuda check" exact
from read_json_auto('results/pr-all/bench2-*/pass-0.jsonl', filename = true);

create or replace temp table r as
select gpu, scenario, batch, vocab, t_ms - clone triton, c_ms - clone cuda, exact from p join clone using (gpu, batch, vocab);

.print '-- summary: geomean change vs. Triton (all, real, FI-eligible), exact cases'
select gpu, count(*) n, bool_and(exact) all_exact,
       round(100 * (exp(avg(ln(cuda / triton))) - 1)) "all",
       round(100 * (exp(avg(ln(cuda / triton)) filter (where scenario like 'real%')) - 1)) "real",
       round(100 * (exp(avg(ln(cuda / triton)) filter (where f.gpu is not null)) - 1)) fi_eligible,
       round(100 * max(cuda / triton - 1)) worst
from r left join fi_ok f using (gpu, scenario, batch, vocab)
group by gpu order by gpu;

.print '-- real rows: CUDA µs and % vs. Triton'
select gpu, scenario, batch, round(triton, 1) triton, round(cuda, 1) cuda, round(100 * (cuda / triton - 1)) pct
from r where scenario like 'real%' order by gpu, scenario, batch;

.print '-- synthetic (B300): geomean change vs. Triton per batch range'
select scenario, case when batch <= 16 then '8-16' when batch <= 128 then '64-128' else '256-2048' end batches,
       round(100 * (exp(avg(ln(cuda / triton))) - 1)) pct
from r where gpu = 'b300' and scenario not like 'real%' group by all order by scenario, min(batch);
