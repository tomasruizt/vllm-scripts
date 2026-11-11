.bail on

CREATE TEMP TABLE raw AS
SELECT regexp_extract(filename, 'case-([0-9]+)', 1)::INTEGER AS case_id, json
FROM read_json_objects(getvariable('renorm_dir') || '/results/timing/case-*.jsonl', filename=true);

CREATE TEMP TABLE metrics AS
SELECT case_id // 30 AS repeat, json ->> 'Scenario' AS scenario, (json ->> 'Batch')::BIGINT AS batch,
       (json ->> 'Vocab')::BIGINT AS vocab, implementation,
       (json ->> (implementation || ' (ms)'))::DOUBLE * 1000 AS api_span_us,
       (json ->> (implementation || ' max TV'))::DOUBLE AS max_tv
FROM raw, unnest(['triton', 'flashinfer-fastpath', 'fi-mask', 'fi-renorm', 'fi-renorm-nondet', 'fi-renorm-probs']) AS names(implementation)
WHERE (json ->> (implementation || ' (ms)')) IS NOT NULL;

COPY metrics TO (getvariable('renorm_dir') || '/metrics.parquet') (FORMAT PARQUET);
COPY (SELECT * FROM read_json_auto(getvariable('renorm_dir') || '/results/validation.jsonl', union_by_name=true))
TO (getvariable('renorm_dir') || '/validation.parquet') (FORMAT PARQUET);
