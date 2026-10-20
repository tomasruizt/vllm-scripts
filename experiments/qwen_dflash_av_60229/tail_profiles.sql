CREATE TEMP TABLE curves AS
WITH arms AS (
    SELECT 'slot_only' AS arm, * FROM read_json_auto('results/qwen-av-tail-profile-20261019/slot_only.json')
    UNION ALL
    SELECT 'mixed' AS arm, * FROM read_json_auto('results/qwen-av-tail-profile-20261019/mixed.json')
), points AS (
    SELECT arm, unnest(verify_curve) AS point FROM arms
)
SELECT arm, point[1]::INTEGER AS tokens, point[2] AS gpu_ms FROM points;

COPY (SELECT policy, tokens, repeat, gpu_ms, 'probe' AS scope, batch FROM read_json_auto('results/qwen-av-tail-profile-20261019/timings.json')
      UNION ALL BY NAME SELECT arm AS policy, tokens, gpu_ms, 'native_curve' AS scope FROM curves)
TO 'results/qwen-av-tail-profile-20261019/metrics.parquet' (FORMAT PARQUET);
COPY (SELECT * FROM read_json_auto('results/qwen-av-tail-profile-20261019/kernels.json'))
TO 'results/qwen-av-tail-profile-20261019/kernels.parquet' (FORMAT PARQUET);
COPY (SELECT * FROM curves ORDER BY arm, tokens)
TO 'results/qwen-av-tail-profile-20261019/curves.csv' (HEADER);
