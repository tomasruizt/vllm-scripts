COPY (
    SELECT a.fold, a.repeat, a.concurrency,
           a.throughput AS native_throughput, b.throughput AS oracle_throughput,
           100 * (b.throughput / a.throughput - 1) AS gain_pct,
           a.aiperf_throughput AS native_aiperf_throughput, b.aiperf_throughput AS oracle_aiperf_throughput,
           100 * (b.aiperf_throughput / a.aiperf_throughput - 1) AS aiperf_gain_pct,
           a.mean_tpot_ms AS native_tpot_ms, b.mean_tpot_ms AS oracle_tpot_ms,
           a.mean_itl_ms AS native_itl_ms, b.mean_itl_ms AS oracle_itl_ms,
           a.acceptance_length AS native_al, b.acceptance_length AS oracle_al
    FROM '__OUTPUT__/metrics.parquet' a JOIN '__OUTPUT__/metrics.parquet' b
      USING (fold, repeat, concurrency)
    WHERE a.mode = 'native' AND b.mode = 'oracle' AND NOT a.traced AND NOT b.traced
      AND a.valid AND b.valid
) TO '__OUTPUT__/pairs.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

COPY (SELECT * FROM '__OUTPUT__/pairs.parquet' ORDER BY concurrency, fold, repeat)
TO '__OUTPUT__/pairs.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    SELECT concurrency, count(*) AS pairs,
           median(native_throughput) AS native_throughput,
           median(oracle_throughput) AS oracle_throughput,
           median(gain_pct) AS paired_gain_pct, min(gain_pct) AS min_gain_pct, max(gain_pct) AS max_gain_pct,
           median(aiperf_gain_pct) AS paired_aiperf_gain_pct,
           median(native_tpot_ms) AS native_tpot_ms, median(oracle_tpot_ms) AS oracle_tpot_ms,
           median(native_itl_ms) AS native_itl_ms, median(oracle_itl_ms) AS oracle_itl_ms,
           median(native_al) AS native_al, median(oracle_al) AS oracle_al
    FROM '__OUTPUT__/pairs.parquet' GROUP BY concurrency ORDER BY concurrency
) TO '__OUTPUT__/comparison.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH outputs AS (
        SELECT fold, repeat, concurrency, mode,
               j.value ->> 'conversation_id' AS id, j.value ->> 'response_text' AS response
        FROM '__OUTPUT__/metrics.parquet' m, json_each(m.aiperf_outputs_json, '$.data') j
        WHERE m.valid AND NOT m.traced
    )
    SELECT a.fold, a.repeat, a.concurrency, count(*) AS compared_requests,
           count(*) FILTER (WHERE a.response = b.response) AS identical_responses
    FROM outputs a JOIN outputs b USING (fold, repeat, concurrency, id)
    WHERE a.mode = 'native' AND b.mode = 'oracle'
    GROUP BY ALL ORDER BY a.fold, a.repeat, a.concurrency
) TO '__OUTPUT__/output-check.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH outputs AS (
        SELECT fold, repeat, concurrency, mode,
               j.value ->> 'conversation_id' AS id, j.value ->> 'response_text' AS response
        FROM '__OUTPUT__/metrics.parquet' m, json_each(m.aiperf_outputs_json, '$.data') j
        WHERE m.valid AND NOT m.traced
    )
    SELECT a.mode, a.repeat, a.concurrency, count(*) AS compared_requests,
           count(*) FILTER (WHERE a.response = b.response) AS identical_responses
    FROM outputs a JOIN outputs b USING (repeat, concurrency, mode, id)
    WHERE a.fold = 'a' AND b.fold = 'b'
    GROUP BY ALL ORDER BY a.mode, a.repeat, a.concurrency
) TO '__OUTPUT__/between-gpu-output-check.csv' (FORMAT CSV, HEADER TRUE);

-- Decision traces
COPY (
    WITH records AS (
        SELECT regexp_extract(filename, 'policy-([ab])-', 1) AS fold,
               regexp_extract(phase, '-c([0-9]+)', 1)::INT AS concurrency, *
        FROM read_json_auto('results/qwen-av-cost-oracle-policy-*-20261019/cost-steps.jsonl',
                            filename=true, union_by_name=true, maximum_object_size=268435456)
        WHERE phase LIKE 'trace-%'
    )
    SELECT r.*, r.time_ns >= m.window_start_ns AND r.time_ns < m.window_end_ns AS steady
    FROM records r JOIN '__OUTPUT__/metrics.parquet' m USING (fold, phase)
) TO '__OUTPUT__/decisions.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

COPY (
    SELECT fold, policy, concurrency, steady, count(*) AS steps,
           count(*) FILTER (WHERE capacity > 0) AS choice_steps,
           count(*) FILTER (WHERE capacity > 0 AND cost_source IN ('measured', 'measured_available')) AS covered_choices,
           count(*) FILTER (WHERE capacity > 0 AND cost_source = 'native_missing_curve') AS missing_curve_choices,
           count(*) FILTER (WHERE capacity > 0 AND cost_source = 'native_mixed_or_incomplete') AS mixed_or_incomplete_choices,
           count(*) FILTER (WHERE cost_source = 'measured' AND budget != native_budget) AS changed_choices,
           count(*) FILTER (WHERE mean_budget IS NOT NULL AND mean_budget != median_budget) AS mean_median_disagreements,
           count(*) FILTER (WHERE cost_source IN ('measured', 'measured_available') AND budget != num_draft_tokens) AS budget_mismatches,
           count(*) FILTER (WHERE policy = 'native' AND budget != native_budget) AS changed_native_budgets,
           avg(budget::DOUBLE / num_reqs) FILTER (WHERE eligible AND num_reqs = concurrency) AS full_decode_drafts_per_req
    FROM '__OUTPUT__/decisions.parquet' GROUP BY ALL ORDER BY fold, concurrency, policy, steady
) TO '__OUTPUT__/decisions.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH utilities AS (
        SELECT d.fold, d.policy, d.concurrency, d.steady,
               d.estimated_accepted_tokens[d.native_budget + 1] AS native_tokens,
               d.estimated_accepted_tokens[d.mean_budget + 1] AS mean_tokens,
               d.estimated_accepted_tokens[d.median_budget + 1] AS median_tokens,
               n.mean_ms AS native_ms, m.mean_ms AS mean_ms, q.mean_ms AS median_ms
        FROM '__OUTPUT__/decisions.parquet' d
        JOIN 'results/cost-oracle-analysis-20261019/points-frozen.parquet' n
          ON n.fold = 'pooled' AND n.num_reqs = d.num_reqs AND n.budget = d.native_budget
        JOIN 'results/cost-oracle-analysis-20261019/points-frozen.parquet' m
          ON m.fold = 'pooled' AND m.num_reqs = d.num_reqs AND m.budget = d.mean_budget
        JOIN 'results/cost-oracle-analysis-20261019/points-frozen.parquet' q
          ON q.fold = 'pooled' AND q.num_reqs = d.num_reqs AND q.budget = d.median_budget
        WHERE d.capacity > 0 AND d.mean_budget IS NOT NULL
    )
    SELECT fold, policy, concurrency, steady, count(*) AS covered_choices,
           median(100 * ((mean_tokens / mean_ms) / (native_tokens / native_ms) - 1)) AS median_objective_gain_pct,
           100 * ((sum(mean_tokens) / sum(mean_ms)) / (sum(native_tokens) / sum(native_ms)) - 1) AS frozen_state_replay_gain_pct,
           median(100 * ((mean_tokens / mean_ms) / (median_tokens / median_ms) - 1)) AS median_statistic_regret_pct,
           quantile_cont(100 * ((mean_tokens / mean_ms) / (median_tokens / median_ms) - 1), .9) AS p90_statistic_regret_pct
    FROM utilities GROUP BY ALL ORDER BY fold, concurrency, policy, steady
) TO '__OUTPUT__/frozen-state-objective.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH aligned AS (
        SELECT *, lead(gap_ms) OVER w AS actual_ms, lead(num_reqs) OVER w AS next_r,
               lead(prefill_tokens) OVER w AS next_prefill, lead(next_num_reqs) OVER w AS next_next_r,
               lead(next_prefill_tokens) OVER w AS next_next_prefill,
               lead(phase) OVER w AS timing_phase, lead(sequence) OVER w AS timing_sequence
        FROM '__OUTPUT__/decisions.parquet' WINDOW w AS (PARTITION BY fold ORDER BY sequence)
    )
    SELECT fold, policy, concurrency, count(*) AS steps,
           median(actual_ms - measured_mean_ms) AS measured_bias_ms,
           avg(actual_ms - measured_mean_ms) AS measured_mean_bias_ms,
           median(abs(actual_ms - measured_mean_ms)) AS measured_median_abs_error_ms,
           quantile_cont(abs(actual_ms - measured_mean_ms), .9) AS measured_p90_abs_error_ms,
           median(actual_ms - selected_cost_ms) AS selected_price_bias_ms
    FROM aligned
    WHERE steady AND eligible AND num_reqs = concurrency AND prefill_tokens = 0
      AND next_prefill = 0 AND next_next_prefill = 0 AND next_r = num_reqs AND next_next_r = num_reqs
      AND timing_sequence = sequence + 1 AND timing_phase = phase AND actual_ms > 0 AND actual_ms < 100
      AND measured_mean_ms IS NOT NULL
    GROUP BY ALL ORDER BY fold, concurrency, policy
) TO '__OUTPUT__/cost-transfer.csv' (FORMAT CSV, HEADER TRUE);
