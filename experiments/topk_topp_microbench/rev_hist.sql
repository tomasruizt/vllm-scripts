-- rev_hist.py runs to one long-format parquet, then a paired summary per run and case.
-- Usage: RUN=name duckdb < rev_hist.sql (reads results/rev-hist/*/times.jsonl and check.jsonl, RUN='' for all)
create or replace table long as
select regexp_extract(filename, 'rev-hist/([^/]+)/', 1) as run, * exclude (filename)
from read_json('results/rev-hist/*/times.jsonl', filename = true, union_by_name = true);
alter table long add column if not exists proc bigint;  -- older runs: one process
update long set proc = 0 where proc is null;
copy (from long order by all) to 'results/rev-hist/results.parquet' (format parquet);

create or replace table checks as
select regexp_extract(filename, 'rev-hist/([^/]+)/', 1) as run, * exclude (filename)
from read_json('results/rev-hist/*/check.jsonl', filename = true, union_by_name = true);
select run, count(*) as checks, count_if(equal) as equal, count_if(pr_close) as pr_exact, count_if(rev_close) as rev_exact
from checks where run = getenv('RUN') or getenv('RUN') = '' group by all order by all;

-- Paired per pass, then the median per process, then the mean over processes (half of them with
-- rev in the first slot, which cancels the slot bias) and the mean per slot order.
alter table long add column if not exists swap boolean;
update long set swap = false where swap is null;
create or replace table paired as
select run, proc, swap, suite, "case", pass_, max(us) filter (impl = 'pr') as pr, max(us) filter (impl = 'rev') as rev
from long group by all;
create or replace table per_proc as
select run, proc, swap, suite, "case", median(pr) as pr, median(rev) as rev, median(rev - pr) as delta,
       100 * median(rev / pr - 1) as pct
from paired group by all;
create or replace table summary as
select run, suite, "case", count(*) as procs, avg(pr) as pr_us, avg(rev) as rev_us, avg(delta) as delta_us,
       avg(pct) as pct, avg(pct) filter (not swap) as pct_pr_first, avg(pct) filter (swap) as pct_rev_first
from per_proc group by all;
.mode markdown
select run, suite, "case", procs, round(pr_us, 1) as pr_us, round(rev_us, 1) as rev_us, round(delta_us, 1) as delta_us,
       round(pct, 1) as pct, round(pct_pr_first, 1) as pct_pr_first, round(pct_rev_first, 1) as pct_rev_first
from summary where run = getenv('RUN') or getenv('RUN') = '' order by run, suite, "case";
