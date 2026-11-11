-- Triton core prototype vs. Triton, CUDA and the FI fast path. Kernel time = call time
-- minus the input clone (clone-only, from results/fiopt6). Usage:
-- duckdb -cmd "SET VARIABLE run='results/triton-core-dev/d4.jsonl'; SET VARIABLE gpu='b300';" < triton_core.sql
create or replace temp table clone as
select Batch, Vocab, median(1000 * "clone-only (ms)") clone
from read_json_auto('results/fiopt6/' || getvariable('gpu') || '/*.jsonl', union_by_name = true)
group by all;

create or replace temp table t as
select r.Scenario scenario, r.Batch batch, r.Vocab vocab,
       1000 * "triton (ms)" - clone triton, 1000 * "triton-core (ms)" - clone core,
       1000 * "cuda (ms)" - clone cuda, 1000 * "fi-opt (ms)" - clone fi,
       "triton-core check" core_ok, "triton check" triton_ok, "cuda check" cuda_ok
from read_json_auto(getvariable('run'), union_by_name = true) r join clone using (Batch, Vocab);

.mode markdown
select 'summary' what, count(*) cases,
       round(100 * (exp(avg(ln(core / triton))) - 1)) "core vs triton %",
       round(100 * (exp(avg(ln(cuda / triton))) - 1)) "cuda vs triton %",
       count(*) filter (core < 0.9 * triton) "core >=10% faster than triton",
       count(*) filter (core > 1.1 * triton) "core >=10% slower than triton",
       count(*) filter (core < 0.9 * cuda) "core >=10% faster than cuda",
       count(*) filter (core > 1.1 * cuda) "core >=10% slower than cuda",
       count(*) filter (core_ok) "core exact", count(*) filter (triton_ok) "triton exact"
from t;

select scenario, round(100 * (exp(avg(ln(core / triton))) - 1)) "core vs triton %",
       round(100 * (exp(avg(ln(cuda / triton))) - 1)) "cuda vs triton %",
       round(100 * (exp(avg(ln(core / cuda))) - 1)) "core vs cuda %",
       count(*) filter (core_ok) || '/' || count(*) "core exact"
from t group by all order by scenario;

select scenario, vocab // 1000 "V k", batch "rows", round(triton, 1) "Triton µs", round(fi, 1) "FI µs",
       round(cuda, 1) "CUDA µs", round(core, 1) "core µs", core_ok exact
from t where scenario like 'real%' or batch in (1, 16, 256, 2048)
order by scenario, vocab, batch;
