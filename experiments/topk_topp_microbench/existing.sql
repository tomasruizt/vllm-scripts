.bail on

CREATE TEMP VIEW wide AS
SELECT split_part(parse_filename(filename, true), '-r', 1) AS workload,
       try_cast(split_part(parse_filename(filename, true), '-r', 2) AS INTEGER) AS repeat,
       * EXCLUDE (filename)
FROM read_json_auto(getvariable('existing_dir') || '/results/*.jsonl', filename=true);

CREATE TEMP VIEW long_metrics AS
SELECT workload, repeat, "Scenario" AS scenario, "Batch" AS batch, "Vocab" AS vocab,
       'triton' AS implementation, "triton (ms)" * 1000 AS api_span_us,
       "triton check" AS reference_close, "triton max TV" AS max_tv
FROM wide
UNION ALL
SELECT workload, repeat, "Scenario", "Batch", "Vocab", 'flashinfer-mask',
       "flashinfer-mask (ms)" * 1000, "flashinfer-mask check", "flashinfer-mask max TV"
FROM wide
UNION ALL
SELECT workload, repeat, "Scenario", "Batch", "Vocab", 'pytorch',
       "pytorch (ms)" * 1000, "pytorch check", "pytorch max TV"
FROM wide;

COPY (SELECT * FROM long_metrics)
TO (getvariable('existing_dir') || '/metrics.parquet') (FORMAT PARQUET);

COPY (
    SELECT workload, scenario, batch, implementation, count(*) AS rounds,
           median(api_span_us) AS median_api_span_us, min(api_span_us) AS min_api_span_us,
           max(api_span_us) AS max_api_span_us, bool_and(reference_close) AS reference_close,
           max(max_tv) AS max_tv
    FROM long_metrics GROUP BY ALL ORDER BY workload, scenario, batch, implementation
) TO (getvariable('existing_dir') || '/summary.csv') (HEADER);

SELECT workload, implementation, count(*) AS cells,
       count(*) FILTER (WHERE NOT reference_close) AS reference_differences, max(max_tv) AS max_tv
FROM long_metrics GROUP BY ALL ORDER BY workload, implementation;
