.bail on

CREATE TEMP VIEW plans AS
SELECT key::INTEGER AS case_index, value ->> 'phase' AS phase,
       (value ->> 'repeat')::INTEGER AS repeat, (value ->> 'top_k')::INTEGER AS top_k,
       (value ->> 'top_p')::DOUBLE AS top_p, value ->> 'distribution' AS distribution,
       try_cast(value ->> 'finite' AS INTEGER) AS finite_logits
FROM read_text('results/topk-topp-expanded-b300-20261104/results/plan.json'), json_each(content);

CREATE TEMP VIEW raw AS
SELECT regexp_extract(filename, 'expanded-(b300|h100)', 1) AS hardware,
       regexp_extract(filename, '(topk-topp-expanded-[^/]+)', 1) AS runner,
       regexp_extract(filename, 'case-([0-9]+)', 1)::INTEGER AS case_index,
       * EXCLUDE (filename)
FROM read_json_auto('results/topk-topp-expanded-*/results/case-*.jsonl', filename=true, union_by_name=true)
WHERE regexp_matches(filename, 'expanded-b300-s[012]-20261104/')
   OR (regexp_matches(filename, 'expanded-h100-frk-20261104/') AND regexp_extract(filename, 'case-([0-9]+)', 1)::INTEGER<168)
   OR regexp_matches(filename, 'expanded-h100-frk-r2-20261104/')
   OR regexp_matches(filename, 'expanded-b300-20261104/results/case-000.jsonl$');

CREATE TEMP VIEW wide AS
SELECT * FROM raw JOIN plans USING (case_index);

CREATE TEMP VIEW metrics AS
SELECT hardware, runner, case_index, phase, repeat, distribution, finite_logits, top_k, top_p,
       "Scenario" AS scenario, "Batch" AS batch, "Vocab" AS vocab,
       'triton' AS implementation, "triton (ms)" * 1000 AS api_span_us,
       "triton check" AS reference_close, "triton max TV" AS max_tv
FROM wide
UNION ALL
SELECT hardware, runner, case_index, phase, repeat, distribution, finite_logits, top_k, top_p,
       "Scenario", "Batch", "Vocab", 'flashinfer-mask',
       "flashinfer-mask (ms)" * 1000, "flashinfer-mask check", "flashinfer-mask max TV"
FROM wide
UNION ALL
SELECT hardware, runner, case_index, phase, repeat, distribution, finite_logits, top_k, top_p,
       "Scenario", "Batch", "Vocab", 'pytorch', "pytorch (ms)" * 1000,
       "pytorch check", "pytorch max TV"
FROM wide;

COPY (
    SELECT *, CASE WHEN implementation != 'flashinfer-mask' THEN 'standard'
        WHEN top_k<=128 AND top_k<vocab AND (scenario IN ('topk_whole', 'topk_topp_whole') OR (scenario='mixed_partial' AND batch<3))
        THEN 'compact' ELSE 'dense-fallback' END AS route
    FROM metrics
) TO 'results/topk-topp-expanded-analysis-20261104/metrics.parquet' (FORMAT PARQUET);

COPY (
    SELECT t.* EXCLUDE (implementation, reference_close, max_tv, route),
           f.api_span_us AS flashinfer_us, f.route AS flashinfer_route,
           t.api_span_us / f.api_span_us AS speedup,
           t.api_span_us - f.api_span_us AS saved_us,
           f.reference_close AS flashinfer_reference_close, f.max_tv AS flashinfer_max_tv
    FROM 'results/topk-topp-expanded-analysis-20261104/metrics.parquet' t
    JOIN 'results/topk-topp-expanded-analysis-20261104/metrics.parquet' f
    USING (hardware, runner, case_index, scenario, batch, vocab)
    WHERE t.implementation='triton' AND f.implementation='flashinfer-mask'
) TO 'results/topk-topp-expanded-analysis-20261104/pairs.parquet' (FORMAT PARQUET);

SELECT hardware, implementation, route, count(*) AS cells,
       count(*) FILTER (WHERE NOT reference_close) AS reference_differences, max(max_tv) AS max_tv
FROM 'results/topk-topp-expanded-analysis-20261104/metrics.parquet'
GROUP BY ALL ORDER BY hardware, implementation, route;
