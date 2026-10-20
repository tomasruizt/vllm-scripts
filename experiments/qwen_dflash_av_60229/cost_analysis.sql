COPY (
    WITH steps AS (
        SELECT mode, concurrency, count(*) AS steps,
               avg(num_draft_tokens) AS drafts_per_step,
               avg(num_draft_tokens::DOUBLE / num_reqs) AS drafts_per_req
        FROM '__OUTPUT__/steps.parquet'
        WHERE mode IN ('oncgdiag', 'oncostcgdiag') AND steady AND decode
          AND cell = 'main' AND num_reqs = concurrency
        GROUP BY ALL
    ), budgets AS (
        SELECT mode, concurrency, 100 * avg((budget = max_budget)::INT) AS full_draft_pct
        FROM '__OUTPUT__/budgets.parquet'
        WHERE steady AND NOT has_prefill AND cell = 'main'
          AND num_reqs = concurrency AND max_budget > 0
        GROUP BY ALL
    )
    SELECT * FROM steps JOIN budgets USING (mode, concurrency)
    ORDER BY concurrency, mode
) TO '__OUTPUT__/decisions.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH curves AS (
        SELECT draft_curve, verify_curve FROM read_json_auto('__OUTPUT__/cost-curves.json')
    )
    SELECT mode, concurrency, count(*) AS steps, median(step_ms) AS measured_ms,
           median(draft_curve[num_reqs + 1][2] + verify_curve[num_tokens + 1][2]) AS fitted_ms,
           median(step_ms - draft_curve[num_reqs + 1][2] - verify_curve[num_tokens + 1][2]) AS residual_ms
    FROM '__OUTPUT__/steps.parquet', curves
    WHERE mode IN ('oncgdiag', 'oncostcgdiag') AND steady AND usable AND decode
      AND cell = 'main' AND num_reqs = concurrency
    GROUP BY ALL ORDER BY concurrency, mode
) TO '__OUTPUT__/cost-transfer.csv' (FORMAT CSV, HEADER TRUE);
