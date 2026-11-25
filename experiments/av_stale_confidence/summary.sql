-- Tables of the README, from results/steps.parquet and results/requests.parquet (see stale.py).
-- Warm steps of the valid workloads only. Throughput = sum(tokens) / sum(modeled cost).
.maxrows 300
create view s as
  select *, case when num_reqs <= 16 then '001-16' when num_reqs <= 32 then '017-32' when num_reqs <= 64 then '033-64'
                 when num_reqs <= 128 then '065-128' else '129+' end reqs
  from 'results/steps.parquet' where warm and exclusion = '';
create view v as select * from s where mode = 'verify_all';
create macro vs_fresh(grp) as table
  with a as (select cost_model, grp g, variant, count(*) steps, sum(tokens) / sum(cost_ms) tput from v group by all)
  select cost_model, g, steps, variant,
         round(100 * (tput / max(tput) filter (variant = 'fresh') over (partition by cost_model, g) - 1), 2) pct
  from a;

.print '1. Fidelity and lag: share of steps (%) where the replayed budget equals the logged policy_budget'
pivot (select mode, variant, round(100 * avg((budget = policy_budget)::int), 2) m from s
       where cost_model = 'table' and variant in ('stale', 'fresh', 'lag2', 'lag3') group by all) on variant using max(m);

.print '2. Rows of the stale input: 1.0-filled (new) and filled although seen before (gap)'
select mode, sum(num_rows) row_count, round(100 * sum(num_new) / sum(num_rows), 2) new_pct, sum(num_gap) gap_rows,
       round(100 * avg((num_new > 0)::int), 1) steps_with_new_pct
from s where cost_model = 'table' and variant = 'stale' group by all;

.print '3. Stale vs fresh inputs, per request (Pearson r of the expected accepted drafts, step t-1 vs t)'
select run, min(r) r_min, max(r) r_max from (
  select run, dataset, concurrency, round(corr(expected_fresh, expected_stale), 2) r
  from 'results/requests.parquet' where mode = 'verify_all' and exclusion = '' group by all) group by all order by 1;

.print '4. Stale vs fresh inputs, per batch: mean |change| of the expected accepted drafts and of the sorted survival curve'
select reqs, count(*) steps, round(100 * avg(abs(expected_stale - expected_fresh) / expected_fresh), 1) expected_abs_pct,
       round(100 * (sum(expected_stale) / sum(expected_fresh) - 1), 2) expected_bias_pct,
       round(100 * avg(curve_l1_stale / drafts), 1) curve_abs_per_slot_pct
from v where cost_model = 'table' and variant = 'stale' group by all order by 1;

.print '5. How often and how much B changes (B_stale vs B_fresh)'
with w as (select cost_model, mode, reqs, run, step, drafts, max(budget) filter (variant = 'stale') bs,
                  max(budget) filter (variant = 'fresh') bf from s group by all)
select cost_model, mode, reqs, count(*) steps, round(100 * avg((bs <> bf)::int), 1) differ_pct,
       round(100 * sum(abs(bs - bf)) / sum(drafts), 2) abs_pct_of_drafts, round(100 * sum(bs - bf) / sum(drafts), 2) signed_pct,
       round(100 * quantile_cont(abs(bs - bf) / drafts, 0.9), 1) p90_pct, round(100 * quantile_cont(abs(bs - bf) / drafts, 0.99), 1) p99_pct,
       round(100 * sum(bs) / sum(drafts), 1) verified_pct
from w where cost_model <> 'linear_cheap' group by all order by 1, 2, 3;

.print '6. Throughput vs the fresh budget (%), by running requests'
pivot (select * from vs_fresh(reqs) where variant in ('stale', 'drift_only', 'fresh_fill', 'mean_fill', 'lag2', 'lag3', 'ema_0.3', 'ema_0.05', 'hindsight'))
  on variant in ('stale', 'drift_only', 'fresh_fill', 'mean_fill', 'lag2', 'lag3', 'ema_0.3', 'ema_0.05', 'hindsight') using max(pct) order by cost_model, g;

.print '7. Stale vs fresh throughput (%), table cost, by run and running requests'
pivot (select run, reqs, round(100 * (sum(tokens) filter (variant = 'stale') / sum(cost_ms) filter (variant = 'stale')
                                      / (sum(tokens) filter (variant = 'fresh') / sum(cost_ms) filter (variant = 'fresh')) - 1), 2) pct
       from v where cost_model = 'table' group by all) on reqs using max(pct) order by run;

.print '8. Stale vs fresh throughput (%), table cost, by dataset and concurrency'
select run, dataset, concurrency c, count(*) filter (variant = 'stale') steps, round(avg(num_reqs)) mean_reqs,
       round(100 * (sum(tokens) filter (variant = 'stale') / sum(cost_ms) filter (variant = 'stale')
                    / (sum(tokens) filter (variant = 'fresh') / sum(cost_ms) filter (variant = 'fresh')) - 1), 2) stale_pct
from v where cost_model = 'table' group by all order by 1, 2, 3;

.print '9. Synthetic share of 1.0-filled rows in the fresh input (throughput vs fresh, %)'
pivot (select * from vs_fresh(reqs) where variant like 'fresh_%fill_%pct') on variant using max(pct) order by cost_model, g;

.print '10. Steps where the stale replay misses the logged budget'
select mode, budget = policy_budget matches, count(*) steps, round(avg(num_new), 1) mean_new_rows,
       round(avg(non_draft_total - num_reqs)) mean_prefill_tokens
from s where cost_model = 'table' and variant = 'stale' group by all order by 1, 2;
