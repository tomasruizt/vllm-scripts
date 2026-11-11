.bail on

CREATE TEMP VIEW metrics AS
SELECT regexp_extract(filename, 'case-([0-9]+)', 1)::INTEGER AS repeat,
       json ->> 'Scenario' AS scenario, (json ->> 'Batch')::BIGINT AS batch,
       (json ->> 'Vocab')::BIGINT AS vocab, implementation,
       (json ->> (implementation || ' (ms)'))::DOUBLE * 1000 AS api_span_us,
       (json ->> (implementation || ' check'))::BOOLEAN AS reference_close,
       (json ->> (implementation || ' max TV'))::DOUBLE AS max_tv
FROM read_json_objects(getvariable('reduced_dir') || '/results/case-*.jsonl', filename=true),
     unnest(['triton', 'flashinfer-mask', 'flashinfer-fastpath']) AS names(implementation)
WHERE json ->> (implementation || ' (ms)') IS NOT NULL;

COPY (SELECT * FROM metrics)
TO (getvariable('reduced_dir') || '/metrics.parquet') (FORMAT PARQUET);

CREATE TEMP VIEW pairs AS
SELECT t.repeat, t.scenario, t.batch, t.vocab,
       t.api_span_us AS triton_us, f.api_span_us AS flashinfer_us,
       f.api_span_us - t.api_span_us AS delta_us,
       t.api_span_us / f.api_span_us AS speedup,
       t.reference_close AS triton_reference_close, f.reference_close AS flashinfer_reference_close,
       t.max_tv AS triton_max_tv, f.max_tv AS flashinfer_max_tv,
       CASE WHEN t.scenario IN ('topk_whole', 'topk_topp_whole') THEN 'compact' ELSE 'dense-fallback' END AS flashinfer_route
FROM metrics t JOIN metrics f USING(repeat, scenario, batch, vocab)
WHERE t.implementation='triton' AND f.implementation IN ('flashinfer-mask', 'flashinfer-fastpath');

COPY (SELECT * FROM pairs)
TO (getvariable('reduced_dir') || '/pairs.parquet') (FORMAT PARQUET);

COPY (
    SELECT scenario, batch, vocab, flashinfer_route, count(*) AS repeats,
           median(triton_us) AS triton_us, median(flashinfer_us) AS flashinfer_us,
           median(delta_us) AS delta_us, min(delta_us) AS min_delta_us, max(delta_us) AS max_delta_us,
           median(speedup) AS paired_speedup, bool_and(triton_reference_close) AS triton_reference_close,
           bool_and(flashinfer_reference_close) AS flashinfer_reference_close,
           max(triton_max_tv) AS triton_max_tv, max(flashinfer_max_tv) AS flashinfer_max_tv
    FROM pairs GROUP BY ALL ORDER BY vocab, batch, scenario
) TO (getvariable('reduced_dir') || '/summary.csv') (HEADER);

SELECT implementation, count(*) AS cells, count(*) FILTER(WHERE NOT reference_close) AS reference_differences,
       max(max_tv) AS max_tv FROM metrics GROUP BY ALL ORDER BY implementation;
