CREATE TEMP TABLE raw_costs AS
WITH curves AS (
    SELECT 'base' AS arm, * FROM read_json_auto('results/qwen-gdn-profile-fix-20261019/base.json')
    UNION ALL
    SELECT 'fixed' AS arm, * FROM read_json_auto('results/qwen-gdn-profile-fix-20261019/fixed.json')
), points AS (
    SELECT arm, 'verify' AS scope, unnest(verify_curve) AS point FROM curves
    UNION ALL
    SELECT arm, 'draft' AS scope, unnest(draft_curve) AS point FROM curves
)
SELECT arm, scope, point[1]::INTEGER AS batch_size, point[2] AS gpu_ms FROM points;

COPY raw_costs TO 'results/qwen-gdn-profile-fix-20261019/metrics.parquet' (FORMAT PARQUET);
COPY (SELECT * FROM raw_costs ORDER BY arm, scope, batch_size) TO 'results/qwen-gdn-profile-fix-20261019/raw-costs.csv' (HEADER);
