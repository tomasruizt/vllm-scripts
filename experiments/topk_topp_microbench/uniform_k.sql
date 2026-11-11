.bail on

CREATE TEMP TABLE metrics AS
SELECT regexp_extract(filename, 'case-([0-9]+)', 1)::INTEGER AS repeat,
       json ->> 'Scenario' AS scenario, (json ->> 'Batch')::BIGINT AS batch,
       (json ->> 'Vocab')::BIGINT AS vocab, implementation,
       (json ->> (implementation || ' (ms)'))::DOUBLE * 1000 AS api_span_us,
       (json ->> (implementation || ' check'))::BOOLEAN AS reference_close,
       (json ->> (implementation || ' max TV'))::DOUBLE AS max_tv
FROM read_json_objects(getvariable('ablation_dir') || '/results/case-*.jsonl', filename=true),
     unnest(['triton', 'fi-uniform', 'fi-bound', 'fi-cap256']) AS names(implementation);

COPY metrics TO (getvariable('ablation_dir') || '/metrics.parquet') (FORMAT PARQUET);

CREATE TEMP TABLE paired AS
SELECT repeat, scenario, batch, vocab,
       max(api_span_us) FILTER(WHERE implementation='triton') AS triton_us,
       max(api_span_us) FILTER(WHERE implementation='fi-uniform') AS uniform_us,
       max(api_span_us) FILTER(WHERE implementation='fi-bound') AS bound_us,
       max(api_span_us) FILTER(WHERE implementation='fi-cap256') AS cap256_us
FROM metrics GROUP BY ALL;

COPY paired TO (getvariable('ablation_dir') || '/pairs.parquet') (FORMAT PARQUET);

CREATE TEMP TABLE summary AS
SELECT scenario, vocab, batch, median(triton_us) AS triton_us,
       median(uniform_us) AS uniform_us, median(bound_us) AS bound_us,
       median(cap256_us) AS cap256_us,
       median(bound_us-uniform_us) AS bound_overhead_us,
       min(bound_us-uniform_us) AS min_bound_overhead_us,
       max(bound_us-uniform_us) AS max_bound_overhead_us,
       median(bound_us-triton_us) AS bound_vs_triton_us,
       min(bound_us-triton_us) AS min_bound_vs_triton_us,
       max(bound_us-triton_us) AS max_bound_vs_triton_us
FROM paired GROUP BY ALL ORDER BY vocab, scenario, batch;

COPY summary TO (getvariable('ablation_dir') || '/summary.csv') (HEADER);
SELECT * FROM summary;
SELECT implementation, count(*) AS cells, bool_and(reference_close) AS all_correct,
       max(max_tv) AS max_tv, bool_and(isfinite(api_span_us)) AS finite
FROM metrics GROUP BY ALL;
