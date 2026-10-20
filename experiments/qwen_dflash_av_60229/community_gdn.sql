CREATE TEMP TABLE curves AS
SELECT CASE WHEN filename LIKE '%/community.json' THEN 'community' ELSE 'gdn' END AS arm,
       draft_curve, verify_curve
FROM read_json_auto(['results/qwen-gdn-community-20261019/community.json', 'results/qwen-gdn-community-20261019/gdn.json'], filename=true);

CREATE TEMP TABLE steps AS
SELECT * FROM read_csv_auto('plots/community-gdn-preview/step-predictions.csv');

COPY (
    SELECT 'measured_period' AS scope, 'historical_base' AS arm, mode, num_reqs,
           num_tokens AS tokens, prefill_tokens, cg_mode, step_ms AS duration_ms
    FROM steps
    UNION ALL
    SELECT 'unclamped_prediction', 'community', mode, num_reqs, num_tokens,
           prefill_tokens, cg_mode, "Community PR" FROM steps
    UNION ALL
    SELECT 'unclamped_prediction', 'gdn', mode, num_reqs, num_tokens,
           prefill_tokens, cg_mode, "Community PR + GDN fixes" FROM steps
    UNION ALL
    SELECT 'native_verify', arm, NULL, NULL, point[1], NULL, NULL, point[2]
    FROM (SELECT arm, unnest(verify_curve) AS point FROM curves)
    UNION ALL
    SELECT 'native_draft', arm, NULL, point[1], NULL, NULL, NULL, point[2]
    FROM (SELECT arm, unnest(draft_curve) AS point FROM curves)
) TO 'results/qwen-gdn-community-20261019/metrics.parquet' (FORMAT PARQUET);
