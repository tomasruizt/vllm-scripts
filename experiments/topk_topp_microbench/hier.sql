-- Hierarchical CUDA filter vs. CUDA, Triton and the Triton core path. Kernel time = call
-- time minus the input clone (clone-only, from results/fiopt6). Usage:
-- duckdb -cmd "SET VARIABLE run='results/hier-cuda/b300.jsonl'; SET VARIABLE gpu='b300';" < hier.sql
create or replace temp table clone as
select Batch, Vocab, median(1000 * "clone-only (ms)") clone
from read_json_auto('results/fiopt6/' || getvariable('gpu') || '/*.jsonl', union_by_name = true)
group by all;

create or replace temp table raw as
select * from read_json_auto(getvariable('run'), union_by_name = true);

create or replace temp table t as
select r.Scenario scenario, r.Batch batch, r.Vocab vocab,
       median(1000 * "hier (ms)" - clone) hier, median(1000 * "cuda (ms)" - clone) cuda,
       median(1000 * "triton (ms)" - clone) triton, median(1000 * "triton-core (ms)" - clone) core,
       median(1000 * "fi-opt (ms)" - clone) fi,
       bool_and("hier check") hier_ok, bool_and("cuda check") cuda_ok
from raw r join clone using (Batch, Vocab) group by all;

.mode markdown
select scenario, count(*) cases, count(*) filter (hier_ok) "hier exact",
       round(exp(avg(ln(hier / cuda))), 2) "hier/cuda", round(exp(avg(ln(hier / triton))), 2) "hier/triton",
       round(exp(avg(ln(hier / core))), 2) "hier/core",
       count(*) filter (hier > 1.1 * cuda) "hier >=10% slower than cuda"
from t group by all order by scenario;

select scenario, batch "rows", round(triton, 1) "Triton µs", round(core, 1) "core µs", round(fi, 1) "FI µs",
       round(cuda, 1) "CUDA µs", round(hier, 1) "hier µs", hier_ok exact
from t where scenario like 'real%' order by scenario, batch;
