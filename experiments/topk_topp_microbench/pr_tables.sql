-- PR #66521 description tables from 3 passes of benchmark_topk_topp.py (triton vs. cuda).
-- Times include the input clone, medians over passes. Batch-1 *_partial cases are left out.
-- Usage: duckdb -cmd "SET VARIABLE dir='results/pr-all/b300-1';" < pr_tables.sql
create or replace temp table p as
select Scenario scenario, Vocab vocab, Batch batch, 1000 * "triton (ms)" triton, 1000 * "cuda (ms)" cuda,
       "cuda check" cuda_ok, "triton check" triton_ok
from read_json_auto(getvariable('dir') || '/pass-*.jsonl', union_by_name = true)
where not (Batch = 1 and Scenario like '%partial');

create or replace temp table m as
select scenario, vocab, batch, median(triton) triton, median(cuda) cuda,
       bool_and(cuda < 0.9 * triton) faster, bool_and(cuda > 1.1 * triton) slower,
       bool_and(cuda_ok) cuda_ok, bool_and(triton_ok) triton_ok
from p group by all;

.mode markdown
select count(*) filter (faster) || ' / ' || count(*) "faster (≥10%)", count(*) filter (slower) || ' / ' || count(*) "slower (≥10%)",
       printf('%.0f%%', 100 * avg(cuda / triton - 1)) "mean change", printf('%.1f', avg(cuda - triton)) "mean Δ µs",
       printf('%.1f', min(cuda - triton)) "best Δ µs", printf('%+.1f', max(cuda - triton)) "worst Δ µs",
       count(*) filter (cuda_ok) "cuda exact", count(*) filter (triton_ok) "triton exact"
from m;

select scenario, printf('%.0f%%', 100 * min(cuda / triton - 1)) best, printf('%.0f%%', 100 * max(cuda / triton - 1)) worst
from m where scenario like 'real%' group by all order by all;

.mode list
.headers off
select '| ' || scenario || ' | ' || vocab || ' | ' || string_agg(coalesce(pc, ''), ' | ' order by b) || ' |'
from (select sv.scenario, sv.vocab, b, printf('%+.0f%%', 100 * (cuda / triton - 1)) pc
      from (select unnest([1, 4, 8, 16, 64, 128, 256, 512, 1024, 2048]) b) bs
      cross join (select distinct scenario, vocab from m) sv
      left join m on m.scenario = sv.scenario and m.vocab = sv.vocab and m.batch = bs.b)
group by scenario, vocab
order by vocab, scenario like 'real%', scenario;

select '| ' || scenario || ' | ' || vocab || ' | ' || batch || ' | ' || printf('%.1f', triton) || ' | ' || printf('%.1f', cuda)
       || ' | ' || printf('%+.0f%%', 100 * (cuda / triton - 1)) || ' |'
from m order by vocab, scenario like 'real%', scenario, batch;
