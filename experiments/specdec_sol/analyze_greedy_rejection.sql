COPY (
    SELECT * FROM read_json_auto(getvariable('input_glob'))
) TO (getvariable('output_dir') || '/metrics.parquet') (FORMAT PARQUET, COMPRESSION ZSTD);

CREATE TEMP VIEW measurements AS
SELECT * FROM read_parquet(getvariable('output_dir') || '/metrics.parquet');

CREATE TEMP VIEW paired AS
SELECT gpu, scenario, requests, target_rows, accepted, repeat,
       max(us) FILTER (WHERE arm='main') AS main_us,
       max(us) FILTER (WHERE arm='pr') AS pr_us
FROM measurements GROUP BY ALL;

CREATE TEMP VIEW cases AS
SELECT gpu, scenario, requests, target_rows, accepted,
       median(main_us) AS main_us, median(pr_us) AS pr_us
FROM paired GROUP BY ALL;

COPY (
    SELECT gpu, scenario, requests, target_rows, accepted,
           main_us, pr_us, main_us-pr_us AS saved_us,
           100*(1-pr_us/main_us) AS reduction_pct
    FROM cases ORDER BY gpu, scenario, requests, target_rows, accepted
) TO (getvariable('output_dir') || '/cases.csv') (HEADER);

COPY (
    SELECT gpu, scenario, requests, target_rows,
           round(min(main_us),2) AS main_min_us, round(max(main_us),2) AS main_max_us,
           round(min(pr_us),2) AS pr_min_us, round(max(pr_us),2) AS pr_max_us,
           round(min(main_us-pr_us),2) AS saved_min_us, round(max(main_us-pr_us),2) AS saved_max_us,
           round(min(100*(1-pr_us/main_us)),1) AS reduction_min_pct,
           round(max(100*(1-pr_us/main_us)),1) AS reduction_max_pct,
           round(avg(pr_us-main_us),2) AS avg_delta_us
    FROM cases GROUP BY ALL ORDER BY gpu, scenario, requests, target_rows
) TO (getvariable('output_dir') || '/summary.csv') (HEADER);

COPY (
    SELECT gpu, scenario, requests, target_rows, accepted,
           round(min(main_us),2) AS main_min_us, round(max(main_us),2) AS main_max_us,
           round(min(pr_us),2) AS pr_min_us, round(max(pr_us),2) AS pr_max_us,
           round(min(100*(1-pr_us/main_us)),1) AS paired_reduction_min_pct,
           round(max(100*(1-pr_us/main_us)),1) AS paired_reduction_max_pct
    FROM paired GROUP BY ALL ORDER BY gpu, scenario, requests, target_rows, accepted
) TO (getvariable('output_dir') || '/repeat-ranges.csv') (HEADER);

SELECT gpu, scenario, count(*) AS measurements,
       bool_and(token_ids_match AND counts_match) AS correctness_passed
FROM measurements GROUP BY ALL ORDER BY gpu, scenario;

SELECT gpu, scenario,
       round(min(main_us-pr_us),2) AS saved_min_us, round(max(main_us-pr_us),2) AS saved_max_us,
       round(min(100*(1-pr_us/main_us)),1) AS reduction_min_pct,
       round(max(100*(1-pr_us/main_us)),1) AS reduction_max_pct
FROM cases GROUP BY ALL ORDER BY gpu, scenario;
