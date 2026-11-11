-- Per GPU and case: mean PR vs. fix µs over passes, delta and %, exactness. Run in results/fallback-balance.
SELECT gpu, suite, source, layout, case_name,
       round(avg(us) FILTER (impl = 'pr'), 1) AS pr_us,
       round(avg(us) FILTER (impl = 'fix'), 1) AS fix_us,
       round(fix_us - pr_us, 1) AS delta_us,
       round(100 * (fix_us / pr_us - 1), 1) AS pct,
       bool_and(exact) AS exact, bool_and(masks_equal) AS equal
FROM 'results.parquet'
GROUP BY ALL
ORDER BY gpu, suite, source, layout, case_name;
