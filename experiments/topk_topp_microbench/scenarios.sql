.bail on

CREATE TEMP TABLE metrics AS
SELECT regexp_extract(filename, 'case-([0-9]+)', 1)::INTEGER // 7 AS repeat,
       json ->> 'Scenario' AS scenario, (json ->> 'Batch')::BIGINT AS batch,
       (json ->> 'Vocab')::BIGINT AS vocab, implementation,
       (json ->> (implementation || ' (ms)'))::DOUBLE * 1000 AS api_span_us,
       (json ->> (implementation || ' check'))::BOOLEAN AS reference_close,
       (json ->> (implementation || ' max TV'))::DOUBLE AS max_tv
FROM read_json_objects(getvariable('scenario_dir') || '/results/case-*.jsonl', filename=true),
     unnest(['triton', 'flashinfer-fastpath']) AS names(implementation);

COPY metrics TO (getvariable('scenario_dir') || '/metrics.parquet') (FORMAT PARQUET);

CREATE TEMP TABLE pairs AS
SELECT t.repeat, t.scenario, t.batch, t.vocab, t.api_span_us AS triton_us,
       f.api_span_us AS flashinfer_us, f.api_span_us - t.api_span_us AS delta_us,
       t.api_span_us / f.api_span_us AS speedup,
       t.reference_close AS triton_reference_close, f.reference_close AS flashinfer_reference_close,
       t.max_tv AS triton_max_tv, f.max_tv AS flashinfer_max_tv
FROM metrics t JOIN metrics f USING(repeat, scenario, batch, vocab)
WHERE t.implementation='triton' AND f.implementation='flashinfer-fastpath';

COPY pairs TO (getvariable('scenario_dir') || '/pairs.parquet') (FORMAT PARQUET);

CREATE TEMP TABLE summary AS
SELECT scenario, vocab, batch, count(*) AS repeats,
       median(triton_us) AS triton_us, median(flashinfer_us) AS flashinfer_us,
       median(delta_us) AS delta_us, min(delta_us) AS min_delta_us, max(delta_us) AS max_delta_us,
       median(speedup) AS paired_speedup,
       bool_and(triton_reference_close AND flashinfer_reference_close) AS all_reference_close,
       max(greatest(triton_max_tv, flashinfer_max_tv)) AS max_tv,
       CASE WHEN max(delta_us / triton_us) <= -0.1 THEN 'win'
            WHEN min(delta_us / triton_us) >= 0.1 THEN 'loss'
            ELSE 'uncertain' END AS outcome
FROM pairs GROUP BY ALL ORDER BY vocab, scenario, batch;

COPY summary TO (getvariable('scenario_dir') || '/summary.csv') (HEADER);
COPY summary TO (getvariable('scenario_dir') || '/summary.parquet') (FORMAT PARQUET);

SELECT implementation, count(*) AS cells, bool_and(reference_close) AS all_reference_close,
       max(max_tv) AS max_tv, bool_and(isfinite(api_span_us)) AS finite
FROM metrics GROUP BY ALL;
SELECT scenario, count(*) AS cases, count(*) FILTER(WHERE outcome='win') AS wins,
       count(*) FILTER(WHERE outcome='loss') AS losses,
       count(*) FILTER(WHERE outcome='uncertain') AS uncertain
FROM summary GROUP BY ALL ORDER BY scenario;
