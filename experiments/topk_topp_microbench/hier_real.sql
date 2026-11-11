-- Real rows: kernel time (call minus clone) at the requested batch sizes, the later run
-- (real2.jsonl, split policy capped by co-resident clusters) replacing real.jsonl rows.
-- duckdb -cmd "SET VARIABLE gpu='b300';" < hier_real.sql
create or replace temp table clone as
select Batch, Vocab, median(1000 * "clone-only (ms)") clone
from read_json_auto('results/fiopt6/' || getvariable('gpu') || '/*.jsonl', union_by_name = true) group by all;
create or replace temp table raw as
select *, 1 run from read_json_auto('results/hier-cuda/' || getvariable('gpu') || '/real.jsonl', union_by_name = true)
union all by name
select *, 2 run from read_json_auto('results/hier-cuda/' || getvariable('gpu') || '/real2.jsonl', union_by_name = true);
create or replace temp table t as
select Scenario scenario, Batch batch, median(1000 * "hier (ms)" - clone) hier,
       median(1000 * "cuda (ms)" - clone) cuda, median(1000 * "triton (ms)" - clone) triton,
       median(1000 * "triton-core (ms)" - clone) core, bool_and("hier check") ok
from (from raw qualify run = max(run) over (partition by Scenario, Batch)) join clone using (Batch, Vocab)
where Batch in (8, 16, 64, 128, 256, 1024, 2048) group by all;
.mode markdown
select scenario, batch "rows", round(triton, 1) "Triton µs", round(core, 1) "New Triton µs", round(cuda, 1) "CUDA µs",
       round(hier, 1) "hier µs", ok exact from t order by scenario, batch;
select scenario, round(exp(avg(ln(hier / cuda))), 2) "hier/CUDA", round(exp(avg(ln(hier / triton))), 2) "hier/Triton",
       round(exp(avg(ln(hier / core))), 2) "hier/New Triton" from t group by all order by scenario;
