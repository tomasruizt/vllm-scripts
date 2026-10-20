-- Offline reference only: native profile uses default captures, runtime calibration uses extended captures.
CREATE TEMP VIEW native_tables AS
SELECT *, max(gpu_ms) OVER (PARTITION BY arm, scope ORDER BY batch_size) AS table_ms
FROM 'results/qwen-gdn-profile-fix-20261019/metrics.parquet';

COPY (
    WITH points AS (
        SELECT unnest(fit.points) AS point
        FROM read_json_auto('results/fixed-drafter-analysis-20261019/cost-curves.json')
    ), shapes AS (
        SELECT point[1]::INT AS num_reqs, point[2]::INT AS num_tokens, point[3] AS measured_ms
        FROM points
    ), reference AS (
        SELECT arm, shapes.*,
            (SELECT table_ms FROM native_tables n WHERE n.arm = arms.arm AND scope = 'draft'
             AND batch_size >= num_reqs ORDER BY batch_size LIMIT 1)
            + (SELECT table_ms FROM native_tables n WHERE n.arm = arms.arm AND scope = 'verify'
               AND batch_size >= num_tokens ORDER BY batch_size LIMIT 1) AS predicted_ms
        FROM shapes CROSS JOIN (SELECT DISTINCT arm FROM native_tables) arms
    )
    SELECT *, predicted_ms - measured_ms AS residual_ms FROM reference ORDER BY arm, num_reqs, num_tokens
) TO 'results/fixed-drafter-analysis-20261019/native-profile-reference.csv' (FORMAT CSV, HEADER TRUE);

SELECT arm, sqrt(avg(residual_ms * residual_ms)) AS rmse_ms, avg(residual_ms) AS bias_ms
FROM 'results/fixed-drafter-analysis-20261019/native-profile-reference.csv' GROUP BY arm;
