.bail on

CREATE TEMP TABLE metrics AS
SELECT regexp_extract(filename, 'case-([0-9]+)', 1)::INTEGER AS repeat,
       json ->> 'Scenario' AS scenario, (json ->> 'Batch')::BIGINT AS batch,
       (json ->> 'Vocab')::BIGINT AS vocab, implementation,
       (json ->> (implementation || ' (ms)'))::DOUBLE * 1000 AS api_span_us,
       (json ->> (implementation || ' check'))::BOOLEAN AS reference_close,
       (json ->> (implementation || ' max TV'))::DOUBLE AS max_tv
FROM read_json_objects(getvariable('unified_dir') || '/results/case-*.jsonl', filename=true),
     unnest(['triton', 'unified']) AS names(implementation);

COPY metrics TO (getvariable('unified_dir') || '/metrics.parquet') (FORMAT PARQUET);

CREATE TEMP TABLE pairs AS
SELECT t.repeat,t.scenario,t.batch,t.vocab,t.api_span_us AS triton_us,
       u.api_span_us AS unified_us,u.api_span_us-t.api_span_us AS delta_us,
       100*(u.api_span_us/t.api_span_us-1) AS change_pct,
       t.reference_close AS triton_reference_close,u.reference_close AS unified_reference_close,
       t.max_tv AS triton_max_tv,u.max_tv AS unified_max_tv
FROM metrics t JOIN metrics u USING(repeat,scenario,batch,vocab)
WHERE t.implementation='triton' AND u.implementation='unified';

COPY pairs TO (getvariable('unified_dir') || '/pairs.parquet') (FORMAT PARQUET);

COPY (
    SELECT scenario,vocab,batch,count(*) AS repeats,
           median(triton_us) AS triton_us,median(unified_us) AS unified_us,
           median(delta_us) AS delta_us,median(change_pct) AS change_pct,
           min(change_pct) AS min_change_pct,max(change_pct) AS max_change_pct,
           bool_and(unified_reference_close) AS unified_reference_close,
           max(unified_max_tv) AS unified_max_tv,
           CASE WHEN max(change_pct)<=-10 THEN 'win'
                WHEN min(change_pct)>=10 THEN 'loss' ELSE 'uncertain' END AS outcome
    FROM pairs GROUP BY ALL ORDER BY vocab,scenario,batch
) TO (getvariable('unified_dir') || '/summary.csv') (HEADER);

SELECT implementation,count(*) AS cells,bool_and(reference_close) AS correct,
       max(max_tv) AS max_tv,bool_and(isfinite(api_span_us)) AS finite
FROM metrics GROUP BY ALL;
