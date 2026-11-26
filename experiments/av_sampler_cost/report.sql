-- Tables of the README. Input: results/steps.parquet (from analyze.py). Run with `make analyze`.
.mode markdown

-- Steady steps of the main cells: the load is at its target and the cell's JIT warm-up is over.
create view steady as
select *, t - min(t) over (partition by run, tag, budget_mode) as cell_s
from 'results/steps.parquet'
where phase = 'main' and budget is not null
qualify cell_s > 4 and num_reqs >= 0.9 * c;

-- Decode-only steps: a prefill chunk both costs forward time and leaves fewer draft slots, which biases the slopes.
create view decode_only as from steady where num_tokens - num_draft_tokens = num_reqs;

-- Random budgets: the slope over the budget is the price of one more verified slot.
create view prices as
select variant, model, mode, c, count(*) as steps, round(avg(num_reqs)) as reqs, max(num_logits) as max_rows,
  1e3 * regr_slope(head_ms, budget) as head_us,
  1e3 * regr_slope(sampler_ms, budget) as sampler_us,
  1e3 * regr_slope(forward_ms, budget) as forward_us,
  1e3 * regr_slope(rest_ms, budget) as rest_us,
  1e3 * regr_slope(step_ms, budget) as step_us,
  1e3 * regr_slope(modeled_cost_ms, budget) as modeled_us,
  regr_intercept(sample_ms, budget) as sample_fixed_ms,
  regr_r2(sample_ms, budget) as sample_r2,
  avg(host_sample_ms) as host_sample_ms
from decode_only where budget_mode = 'random'
group by all;

copy (from prices) to 'results/prices.parquet';

.print '## Price of one more verified slot, microseconds (random budgets)'
select variant, model, c, mode, steps, reqs, max_rows,
  round(head_us, 1) as "LM head", round(sampler_us, 1) as sampler, round(forward_us, 1) as forward,
  round(rest_us, 1) as rest, round(step_us, 1) as "whole step", round(modeled_us, 1) as modeled,
  round(100 * (head_us + sampler_us) / modeled_us) as "untimed / modeled %",
  round(100 * (head_us + sampler_us) / step_us) as "untimed / whole step %",
  round(100 * modeled_us / step_us) as "modeled / whole step %"
from prices order by variant, model, c, mode;

.print
.print '## Fixed part of sample() and fit quality (random budgets)'
select variant, model, c, mode, round(sample_fixed_ms, 3) as "sample() at budget 0, ms",
  round(sample_r2, 3) as r2, round(host_sample_ms, 3) as "host time in sample(), ms"
from prices order by variant, model, c, mode;

.print
.print '## Policy budgets: measured step time against modeled cost'
select variant, model, c, mode, count(*) as steps, round(avg(num_reqs)) as reqs,
  round(100 * sum(budget) / sum(max_budget)) as "verified %",
  round(avg(modeled_cost_ms), 2) as "modeled ms", round(avg(step_ms), 2) as "step ms",
  round(avg(forward_ms), 2) as "forward ms", round(avg(sample_ms), 2) as "sample() ms",
  round(avg(rest_ms), 2) as "rest ms", round(100 * avg(sample_ms) / avg(step_ms)) as "sample() / step %"
from steady where budget_mode = 'policy'
group by all order by variant, model, c, mode;
