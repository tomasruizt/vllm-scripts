.bail on

COPY (
    SELECT CASE WHEN filename LIKE '%small%' THEN 'small' WHEN filename LIKE '%large%' THEN 'large' ELSE 'captured' END AS phase, *
    FROM read_json_auto([
        'results/topk-topp-validation-small-20261103-r3/results/validation.jsonl',
        'results/topk-topp-validation-large-20261103-r2/results/validation.jsonl',
        'results/topk-topp-validation-captured-20261103/results/validation.jsonl',
        'results/topk-topp-validation-deterministic-small-20261103/results/validation.jsonl',
        'results/topk-topp-validation-deterministic-large-20261103/results/validation.jsonl',
        'results/topk-topp-validation-deterministic-20261103/results/validation.jsonl'
    ], union_by_name=true, filename=true)
) TO 'results/topk-topp-validation-20261103/validation.parquet' (FORMAT PARQUET);

COPY (
    SELECT * FROM read_json_auto([
        'results/topk-topp-validation-verifier-20261103/results/integration.json',
        'results/topk-topp-validation-deterministic-verifier-20261103/results/integration.json'
    ], union_by_name=true, filename=true)
) TO 'results/topk-topp-validation-20261103/integration.parquet' (FORMAT PARQUET);

COPY (
    SELECT * EXCLUDE (kernel_us) FROM read_json_auto([
        'results/topk-topp-validation-perf-20261103/results/metrics.json',
        'results/topk-topp-validation-deterministic-perf-20261103/results/metrics.json'
    ], union_by_name=true, filename=true)
) TO 'results/topk-topp-validation-20261103/performance.parquet' (FORMAT PARQUET);

COPY (
    SELECT filename, batch, vocab, distribution, implementation, "round", 'eager' AS method,
           generate_subscripts(eager_us, 1) AS repeat, unnest(eager_us) AS duration_us
    FROM 'results/topk-topp-validation-20261103/performance.parquet'
    UNION ALL
    SELECT filename, batch, vocab, distribution, implementation, "round", 'cupti-span' AS method,
           generate_subscripts(cupti_us, 1) AS repeat, unnest(cupti_us) AS duration_us
    FROM 'results/topk-topp-validation-20261103/performance.parquet'
) TO 'results/topk-topp-validation-20261103/performance-samples.parquet' (FORMAT PARQUET);

COPY (
    SELECT * FROM read_json_auto('results/topk-topp-validation-explain-20261103/results/explanations.json')
) TO 'results/topk-topp-validation-20261103/counterexamples.parquet' (FORMAT PARQUET);

COPY (
    SELECT * FROM read_json_auto('results/topk-topp-validation-boundary-20261103/results/explanations.json')
) TO 'results/topk-topp-validation-20261103/boundary-counterexamples.parquet' (FORMAT PARQUET);

COPY (
    SELECT phase, backend, route, count(*) AS comparisons,
           count(*) FILTER (WHERE NOT native_exact) AS native_differences,
           count(*) FILTER (WHERE NOT basic_valid) AS basic_failures,
           count(*) FILTER (WHERE NOT ranking_valid) AS ranking_failures,
           max(native_tv) AS max_native_tv
    FROM 'results/topk-topp-validation-20261103/validation.parquet'
    GROUP BY ALL ORDER BY phase, backend, route
) TO 'results/topk-topp-validation-20261103/summary.csv' (HEADER);
