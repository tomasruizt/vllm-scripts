COPY (
    WITH records AS (
        SELECT regexp_extract(filename, '-([ab])-20261019', 1) AS fold,
               list_avg(contexts) AS context_mean, *
        FROM read_json_auto('results/__PREFIX__-[ab]-20261019/cost-steps.jsonl',
                            filename=true, union_by_name=true, maximum_object_size=268435456)
        WHERE phase IS NOT NULL
    ), boundaries AS (
        SELECT *, lag(sequence) OVER (PARTITION BY fold ORDER BY time_ns, sequence) AS previous_sequence
        FROM records
    )
    SELECT * EXCLUDE (previous_sequence),
           sum((previous_sequence IS NOT NULL AND sequence <= previous_sequence)::INT)
             OVER (PARTITION BY fold ORDER BY time_ns, sequence) AS trace_session
    FROM boundaries
) TO '__OUTPUT__/raw-steps.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

COPY (
    WITH aligned AS (
        SELECT *, lead(gap_ms) OVER w AS actual_ms,
               lead(sequence) OVER w AS timing_sequence, lead(phase) OVER w AS timing_phase,
               lead(num_reqs) OVER w AS timing_r, lead(prefill_tokens) OVER w AS timing_prefill,
               lead(next_num_reqs) OVER w AS timing_next_r,
               lead(next_prefill_tokens) OVER w AS timing_next_prefill
        FROM '__OUTPUT__/raw-steps.parquet' WINDOW w AS (PARTITION BY fold, trace_session ORDER BY time_ns, sequence)
    )
    SELECT a.*, m.cost_valid AS cell_cost_valid,
              a.time_ns >= m.cell_start_ns AND a.time_ns < m.cell_end_ns AS within_cell,
              eligible AND prefill_tokens = 0 AND timing_prefill = 0 AND timing_next_prefill = 0
              AND timing_r = num_reqs AND timing_next_r = num_reqs
              AND timing_sequence = sequence + 1 AND timing_phase = phase
              AND actual_ms > 0 AND actual_ms < 100
              AND budget = num_draft_tokens AND num_tokens = num_reqs + budget AS usable
    FROM aligned a LEFT JOIN '__OUTPUT__/metrics.parquet' m USING (fold, phase)
) TO '__OUTPUT__/steps.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

CREATE TEMP VIEW points AS
SELECT fold, num_reqs, budget, max(capacity) AS candidate_capacity, count(*) AS samples, avg(actual_ms) AS mean_ms,
       median(actual_ms) AS median_ms, stddev_samp(actual_ms) AS std_ms,
       quantile_cont(actual_ms, .1) AS p10_ms, quantile_cont(actual_ms, .9) AS p90_ms,
       min(context_mean) AS min_context_mean, max(context_mean) AS max_context_mean
FROM '__OUTPUT__/steps.parquet' WHERE usable AND cell_cost_valid AND within_cell AND policy = 'sweep' GROUP BY ALL
UNION ALL
SELECT 'pooled', num_reqs, budget, max(capacity), count(*), avg(actual_ms), median(actual_ms), stddev_samp(actual_ms),
       quantile_cont(actual_ms, .1), quantile_cont(actual_ms, .9), min(context_mean), max(context_mean)
FROM '__OUTPUT__/steps.parquet' WHERE usable AND cell_cost_valid AND within_cell AND policy = 'sweep' GROUP BY num_reqs, budget;

COPY (SELECT * FROM points ORDER BY fold, num_reqs, budget)
TO '__OUTPUT__/points.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (SELECT * FROM points ORDER BY fold, num_reqs, budget)
TO '__OUTPUT__/points.csv' (FORMAT CSV, HEADER TRUE);

CREATE TEMP VIEW curves AS
SELECT fold, num_reqs, count(*) AS observed_budgets, min(samples) AS minimum_samples,
       sum(samples) AS usable_samples,
       max(candidate_capacity) AS candidate_capacity,
       count(*) = max(candidate_capacity) + 1 AND min(samples) >= 5 AS complete_5
FROM points GROUP BY fold, num_reqs;

COPY (SELECT * FROM curves ORDER BY fold, num_reqs)
TO '__OUTPUT__/coverage.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH grid AS (
        SELECT num_reqs, b.budget FROM curves c
        CROSS JOIN LATERAL range(c.candidate_capacity + 1) b(budget)
        WHERE fold = 'pooled' AND num_reqs IN (254, 255, 256)
    )
    SELECT g.num_reqs, list(g.budget ORDER BY g.budget) AS budgets,
           min(coalesce(p.samples, 0)) AS minimum_samples
    FROM grid g LEFT JOIN points p ON p.fold = 'pooled' AND p.num_reqs = g.num_reqs AND p.budget = g.budget
    WHERE coalesce(p.samples, 0) < 5 GROUP BY g.num_reqs ORDER BY g.num_reqs
) TO '__OUTPUT__/gaps.json' (FORMAT JSON, ARRAY TRUE);

COPY (
    SELECT fold, num_reqs, count(*) AS steps,
           corr(gap_ms, budget) AS raw_budget_correlation,
           corr(actual_ms, budget) AS aligned_budget_correlation
    FROM '__OUTPUT__/steps.parquet' WHERE usable AND policy = 'sweep'
    GROUP BY ALL ORDER BY fold, num_reqs
) TO '__OUTPUT__/timing-alignment.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH records AS (
        SELECT regexp_extract(filename, '-([ab])-20261019', 1) AS fold,
               contains(filename, 'rank1') AS rank1, sequence, time_ns, phase, budget, num_draft_tokens, num_reqs, draft_lens, cg_mode
        FROM read_json_auto('results/__PREFIX__-[ab]-20261019/cost-steps*.jsonl',
                            filename=true, union_by_name=true, maximum_object_size=268435456)
    ), boundaries AS (
        SELECT *, lag(sequence) OVER (PARTITION BY fold, rank1 ORDER BY time_ns, sequence) AS previous_sequence
        FROM records
    ), ranks AS (
        SELECT *, sum((previous_sequence IS NOT NULL AND sequence <= previous_sequence)::INT)
          OVER (PARTITION BY fold, rank1 ORDER BY time_ns, sequence) AS trace_session FROM boundaries
    )
    SELECT a.fold, count(*) AS paired_steps,
           count(*) FILTER (WHERE a.phase != b.phase OR a.budget != b.budget
                                 OR a.num_draft_tokens != b.num_draft_tokens OR a.num_reqs != b.num_reqs
                                 OR a.draft_lens != b.draft_lens OR a.cg_mode != b.cg_mode) AS disagreements
    FROM ranks a JOIN ranks b USING (fold, trace_session, sequence) WHERE NOT a.rank1 AND b.rank1 GROUP BY a.fold
) TO '__OUTPUT__/rank-agreement.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    SELECT s.fold, s.phase, s.policy, s.num_reqs, s.prefill_tokens > 0 AS mixed,
           s.capacity, count(*) AS choices, median(s.non_draft_total) AS median_non_draft_total,
           count(*) FILTER (WHERE s.eligible AND c.complete_5) AS priced_choices
    FROM '__OUTPUT__/steps.parquet' s
    JOIN '__OUTPUT__/metrics.parquet' m USING (fold, phase)
    LEFT JOIN curves c ON c.fold = 'pooled' AND s.num_reqs = c.num_reqs
    WHERE s.phase LIKE 'trace-%' AND s.capacity > 0
      AND s.time_ns >= m.window_start_ns AND s.time_ns < m.window_end_ns
    GROUP BY ALL ORDER BY s.fold, choices DESC
) TO '__OUTPUT__/shapes.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    SELECT s.fold, s.num_reqs, count(*) AS steps, count(p.mean_ms) AS priced_steps,
           median(abs(s.actual_ms - p.mean_ms)) AS median_abs_error_ms,
           median(s.actual_ms - p.mean_ms) AS median_bias_ms
    FROM '__OUTPUT__/steps.parquet' s
    LEFT JOIN points p ON p.num_reqs = s.num_reqs AND p.budget = s.budget
      AND p.fold != s.fold AND p.fold IN ('a', 'b') AND p.samples >= 5
    WHERE s.usable AND s.policy = 'native'
    GROUP BY ALL ORDER BY s.fold, s.num_reqs
) TO '__OUTPUT__/validation.csv' (FORMAT CSV, HEADER TRUE);
