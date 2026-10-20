COPY (
    SELECT mode, num_reqs, num_tokens, count(*) AS steps,
           count(*) FILTER (WHERE steady) AS steady_steps,
           median(step_ms) AS all_ms, median(step_ms) FILTER (WHERE steady) AS steady_ms,
           steady_ms - all_ms AS delta_ms
    FROM '__OUTPUT__/steps.parquet'
    WHERE mode LIKE 'oncap%' AND decode AND usable AND cell = 'main' AND num_reqs = concurrency
    GROUP BY ALL HAVING count(*) >= 20 ORDER BY mode, num_reqs, num_tokens
) TO '__OUTPUT__/calibration-windows.csv' (FORMAT CSV, HEADER TRUE);

COPY (
    WITH shapes AS (
        SELECT mode, num_reqs, num_tokens, count(*) AS steps, median(step_ms) AS measured_ms
        FROM '__OUTPUT__/steps.parquet'
        WHERE mode IN ('oncgdiag', 'oncostcgdiag') AND steady AND usable AND decode
          AND cell = 'main' AND num_reqs = concurrency
        GROUP BY ALL HAVING count(*) >= 20
    ), curves AS (
        SELECT draft_curve, verify_curve FROM read_json_auto('__OUTPUT__/cost-curves.json')
    )
    SELECT shapes.*, draft_curve[num_reqs + 1][2] + verify_curve[num_tokens + 1][2] AS fitted_ms,
           measured_ms - fitted_ms AS residual_ms
    FROM shapes, curves ORDER BY num_reqs, mode, num_tokens
) TO '__OUTPUT__/runtime-shapes.csv' (FORMAT CSV, HEADER TRUE);
