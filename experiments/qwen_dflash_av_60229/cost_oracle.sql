COPY (
    SELECT regexp_extract(filename, 'qwen-cost-oracle-([ab])-20261019', 1) AS fold,
           regexp_extract(phase, '-c([0-9]+)', 1)::INT AS concurrency,
           list_avg(contexts) AS context_mean, list_min(contexts) AS context_min,
           list_max(contexts) AS context_max, *,
           eligible AND prefill_tokens = 0 AND next_prefill_tokens = 0
             AND next_num_reqs = num_reqs AND gap_ms > 0 AND gap_ms < 100
             AND budget = num_draft_tokens AND num_tokens = num_reqs + budget AS raw_usable
    FROM read_json_auto('results/qwen-cost-oracle-*-20261019/cost-steps.jsonl',
                        filename=true, union_by_name=true, maximum_object_size=268435456)
    WHERE phase IS NOT NULL
) TO '__OUTPUT__/raw-steps.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

-- record_confidences waits on the preceding batch's double-buffered copy event.
-- Gap t+1 therefore measures batch t, not batch t+1. Require stable decode neighbors.
COPY (
    WITH aligned AS (
        SELECT * EXCLUDE (cost_ms, adjusted_ms, probe_ms),
               lead(cost_ms) OVER w AS cost_ms, lead(adjusted_ms) OVER w AS adjusted_ms,
               lead(probe_ms) OVER w AS probe_ms, lead(sequence) OVER w AS timing_sequence,
               lead(num_reqs) OVER w AS timing_num_reqs, lead(phase) OVER w AS timing_phase,
               lead(prefill_tokens) OVER w AS timing_prefill_tokens,
               lead(next_num_reqs) OVER w AS timing_next_num_reqs,
               lead(next_prefill_tokens) OVER w AS timing_next_prefill_tokens
        FROM '__OUTPUT__/raw-steps.parquet' WINDOW w AS (PARTITION BY fold ORDER BY sequence)
    )
    SELECT *, eligible AND prefill_tokens = 0 AND timing_prefill_tokens = 0
              AND timing_next_prefill_tokens = 0 AND timing_num_reqs = num_reqs
              AND timing_next_num_reqs = num_reqs AND timing_sequence = sequence + 1
              AND timing_phase = phase AND cost_ms > 0 AND cost_ms < 100
              AND budget = num_draft_tokens AND num_tokens = num_reqs + budget AS usable
    FROM aligned
) TO '__OUTPUT__/steps.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

CREATE TEMP VIEW points AS
SELECT fold, num_reqs, budget, count(*) AS samples, median(cost_ms) AS median_ms,
       median(adjusted_ms) AS adjusted_ms,
       quantile_cont(cost_ms, .1) AS p10_ms, quantile_cont(cost_ms, .9) AS p90_ms,
       median(probe_ms) AS probe_ms, min(context_mean) AS min_context_mean,
       max(context_mean) AS max_context_mean, avg(cost_ms) AS mean_ms, stddev_samp(cost_ms) AS std_ms
FROM '__OUTPUT__/steps.parquet' WHERE usable AND policy = 'sweep' GROUP BY ALL
UNION ALL
SELECT 'pooled', num_reqs, budget, count(*), median(cost_ms), median(adjusted_ms),
       quantile_cont(cost_ms, .1), quantile_cont(cost_ms, .9), median(probe_ms),
       min(context_mean), max(context_mean), avg(cost_ms), stddev_samp(cost_ms)
FROM '__OUTPUT__/steps.parquet' WHERE usable AND policy = 'sweep' GROUP BY num_reqs, budget;

COPY (SELECT * FROM points ORDER BY fold, num_reqs, budget)
TO '__OUTPUT__/points.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (SELECT * FROM points ORDER BY fold, num_reqs, budget)
TO '__OUTPUT__/points.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH grid AS (
        SELECT f.fold, r.num_reqs, b.budget FROM (VALUES ('a'), ('b'), ('pooled')) f(fold)
        CROSS JOIN (VALUES (1), (8), (32), (64), (128)) r(num_reqs)
        CROSS JOIN LATERAL range(7 * r.num_reqs + 1) b(budget)
    )
    SELECT g.fold, g.num_reqs, count(*) AS budgets,
           count(p.samples) AS observed, count(*) FILTER (WHERE p.samples >= 10) AS at_least_10,
           min(coalesce(p.samples, 0)) AS minimum_samples,
           sum(coalesce(p.samples, 0)) AS usable_steps
    FROM grid g LEFT JOIN points p USING (fold, num_reqs, budget)
    GROUP BY ALL ORDER BY fold, num_reqs
) TO '__OUTPUT__/coverage.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    SELECT a.num_reqs, count(*) AS paired_budgets,
           median(abs(a.median_ms - b.median_ms)) AS median_fold_difference_ms,
           quantile_cont(abs(a.median_ms - b.median_ms), .9) AS p90_fold_difference_ms,
           median(((a.p90_ms - a.p10_ms) + (b.p90_ms - b.p10_ms)) / 2) AS median_step_spread_ms
    FROM points a JOIN points b USING (num_reqs, budget)
    WHERE a.fold = 'a' AND b.fold = 'b' AND a.samples >= 10 AND b.samples >= 10
    GROUP BY a.num_reqs ORDER BY a.num_reqs
) TO '__OUTPUT__/repeatability.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH shifted AS (
        SELECT *, lag(budget) OVER w AS previous_budget, lead(budget) OVER w AS following_budget
        FROM '__OUTPUT__/raw-steps.parquet' WINDOW w AS (PARTITION BY fold ORDER BY sequence)
    ), grouped AS (
        SELECT fold, num_reqs, budget_offset, b, count(*) AS n, median(cost_ms) AS ms
        FROM shifted CROSS JOIN LATERAL
             (VALUES (-1, previous_budget), (0, budget), (1, following_budget)) v(budget_offset, b)
        WHERE raw_usable AND policy = 'sweep' AND num_reqs IN (1, 8)
          AND b BETWEEN 0 AND 7 * num_reqs GROUP BY ALL HAVING n >= 20
    )
    SELECT fold, num_reqs, budget_offset, count(*) AS budgets,
           max(ms) - min(ms) AS range_ms, stddev_pop(ms) AS between_budget_sd_ms
    FROM grouped GROUP BY ALL ORDER BY fold, num_reqs, budget_offset
) TO '__OUTPUT__/timing-alignment.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH controls AS (
        SELECT num_reqs, num_draft_tokens AS budget, median(step_ms) AS ms
        FROM 'results/fixed-drafter-analysis-20261019/steps.parquet'
        WHERE usable AND decode AND steady AND num_reqs = concurrency
          AND mode IN ('oncap1cgdiag', 'oncap3cgdiag', 'oncap7cgdiag')
          AND num_draft_tokens IN (num_reqs, 3 * num_reqs, 7 * num_reqs) GROUP BY ALL
    )
    SELECT p.fold, p.num_reqs, p.budget, p.samples, p.median_ms AS measured_ms, c.ms AS previous_ms,
           p.median_ms - b.median_ms AS measured_marginal_ms,
           c.ms - cb.ms AS previous_marginal_ms,
           (p.median_ms - b.median_ms) - (c.ms - cb.ms) AS marginal_error_ms
    FROM points p JOIN controls c USING (num_reqs, budget)
    JOIN points b ON p.fold = b.fold AND p.num_reqs = b.num_reqs AND b.budget = b.num_reqs
    JOIN controls cb ON p.num_reqs = cb.num_reqs AND cb.budget = cb.num_reqs
    WHERE p.num_reqs IN (1, 8, 32, 64, 128) ORDER BY p.fold, p.num_reqs, p.budget
) TO '__OUTPUT__/constant-budget-check.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    SELECT v.fold, v.num_reqs, v.budget, v.context_mean, v.cost_ms,
           p.median_ms AS predicted_ms, v.cost_ms - p.median_ms AS residual_ms,
           p.mean_ms AS mean_predicted_ms, v.cost_ms - p.mean_ms AS mean_residual_ms,
           v.adjusted_ms - p.adjusted_ms AS adjusted_residual_ms, p.samples
    FROM '__OUTPUT__/steps.parquet' v LEFT JOIN points p
      ON v.num_reqs = p.num_reqs AND v.budget = p.budget AND v.fold != p.fold AND p.fold IN ('a', 'b')
    WHERE v.usable AND v.policy = 'native'
) TO '__OUTPUT__/validation.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

COPY (
    SELECT fold, num_reqs, count(*) AS steps, count(predicted_ms) AS covered_steps,
           count(*) FILTER (WHERE samples >= 10) AS covered_10,
           median(residual_ms) AS bias_ms, median(abs(residual_ms)) AS median_abs_error_ms,
           quantile_cont(abs(residual_ms), .9) AS p90_abs_error_ms,
           avg(mean_residual_ms) AS mean_prediction_bias_ms
    FROM '__OUTPUT__/validation.parquet' GROUP BY ALL ORDER BY fold, num_reqs
) TO '__OUTPUT__/validation.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    SELECT v.fold, v.concurrency, count(*) AS recorded_steps,
           count(*) FILTER (WHERE v.usable) AS usable_decode_steps,
           count(*) FILTER (WHERE v.usable AND p.samples >= 10) AS supported_decode_steps,
           count(*) FILTER (WHERE v.prefill_tokens > 0) AS mixed_or_prefill_steps,
           count(*) FILTER (WHERE v.budget != v.native_budget) AS changed_native_budgets,
           median(v.probe_ms) AS median_probe_ms,
           quantile_cont(v.probe_ms / v.cost_ms, .9) AS p90_probe_fraction
    FROM '__OUTPUT__/steps.parquet' v LEFT JOIN points p
      ON v.num_reqs = p.num_reqs AND v.budget = p.budget AND v.fold != p.fold AND p.fold IN ('a', 'b')
    WHERE v.policy = 'native' AND v.phase LIKE 'validation-%'
    GROUP BY ALL ORDER BY v.fold, v.concurrency
) TO '__OUTPUT__/native-coverage.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH curves AS (
        SELECT fold, num_reqs, count(*) = 7 * num_reqs + 1 AS complete,
               min(samples) AS minimum_samples FROM points GROUP BY fold, num_reqs
    )
    SELECT v.fold, v.concurrency, count(*) AS recorded_steps,
           count(*) FILTER (WHERE v.usable) AS usable_decode_steps,
           count(*) FILTER (WHERE v.usable AND c.complete) AS other_fold_complete_curve,
           count(*) FILTER (WHERE v.usable AND c.complete AND c.minimum_samples >= 5) AS other_fold_curve_5,
           count(*) FILTER (WHERE v.usable AND c.complete AND c.minimum_samples >= 10) AS other_fold_curve_10,
           count(*) FILTER (WHERE v.usable AND p.complete) AS pooled_complete_curve,
           count(*) FILTER (WHERE v.usable AND p.complete AND p.minimum_samples >= 5) AS pooled_curve_5,
           count(*) FILTER (WHERE v.usable AND p.complete AND p.minimum_samples >= 10) AS pooled_curve_10
    FROM '__OUTPUT__/steps.parquet' v
    LEFT JOIN curves c ON v.num_reqs = c.num_reqs AND v.fold != c.fold AND c.fold IN ('a', 'b')
    LEFT JOIN curves p ON v.num_reqs = p.num_reqs AND p.fold = 'pooled'
    WHERE v.policy = 'native' AND v.phase LIKE 'validation-%'
    GROUP BY ALL ORDER BY v.fold, v.concurrency
) TO '__OUTPUT__/native-curve-coverage.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    SELECT fold, num_reqs, floor(context_mean / 256) * 256 AS context_bin,
           count(*) AS steps, count(predicted_ms) AS covered_steps,
           median(residual_ms) AS bias_ms, median(abs(residual_ms)) AS median_abs_error_ms,
           quantile_cont(abs(residual_ms), .9) AS p90_abs_error_ms,
           median(abs(adjusted_residual_ms)) AS adjusted_median_abs_error_ms
    FROM '__OUTPUT__/validation.parquet' GROUP BY ALL ORDER BY fold, num_reqs, context_bin
) TO '__OUTPUT__/validation-context.csv' (FORMAT CSV, HEADER TRUE);
