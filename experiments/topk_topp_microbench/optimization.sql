.bail on

CREATE TEMP TABLE metrics AS
SELECT regexp_extract(filename, 'case-([0-9]+)', 1)::INTEGER // 3 AS repeat,
       json ->> 'Scenario' AS scenario, (json ->> 'Batch')::BIGINT AS batch,
       (json ->> 'Vocab')::BIGINT AS vocab, implementation,
       (json ->> (implementation || ' (ms)'))::DOUBLE * 1000 AS api_span_us,
       (json ->> (implementation || ' check'))::BOOLEAN AS reference_close,
       (json ->> (implementation || ' max TV'))::DOUBLE AS max_tv
FROM read_json_objects(getvariable('optimization_dir') || '/results/case-*.jsonl', filename=true),
     unnest(['triton', 'fi-baseline', 'fi-no-flip', 'fi-unsorted', 'flashinfer-fastpath']) AS names(implementation);

COPY metrics TO (getvariable('optimization_dir') || '/metrics.parquet') (FORMAT PARQUET);

CREATE TEMP TABLE pairs AS
SELECT repeat, scenario, vocab, batch,
       max(api_span_us) FILTER(WHERE implementation='triton') AS triton_us,
       max(api_span_us) FILTER(WHERE implementation='fi-baseline') AS baseline_us,
       max(api_span_us) FILTER(WHERE implementation='fi-no-flip') AS no_flip_us,
       max(api_span_us) FILTER(WHERE implementation='fi-unsorted') AS unsorted_us,
       max(api_span_us) FILTER(WHERE implementation='flashinfer-fastpath') AS fused_us
FROM metrics GROUP BY ALL;

COPY pairs TO (getvariable('optimization_dir') || '/pairs.parquet') (FORMAT PARQUET);

COPY (
    SELECT scenario,vocab,batch,count(*) AS repeats,
           median(triton_us) AS triton_us,median(baseline_us) AS baseline_us,
           median(no_flip_us) AS no_flip_us,median(unsorted_us) AS unsorted_us,
           median(fused_us) AS fused_us,
           median(fused_us-baseline_us) AS delta_baseline_us,
           median(fused_us-triton_us) AS delta_triton_us,
           min(fused_us-triton_us) AS min_delta_triton_us,
           max(fused_us-triton_us) AS max_delta_triton_us
    FROM pairs GROUP BY ALL ORDER BY vocab,scenario,batch
) TO (getvariable('optimization_dir') || '/summary.csv') (HEADER);

SELECT implementation,count(*) AS cells,bool_and(reference_close) AS correct,
       max(max_tv) AS max_tv,bool_and(isfinite(api_span_us)) AS finite
FROM metrics GROUP BY ALL;
