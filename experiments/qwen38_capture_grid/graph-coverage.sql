-- Run from this experiment directory with duckdb -f graph-coverage.sql.
-- Keep only complete 10-second logging intervals inside the steady window.
SET memory_limit='4GB';
SET threads=2;
CREATE TEMP VIEW requests AS
SELECT regexp_extract(filename, '/(default|2k|4k)/artifacts/', 1) AS grid,
       regexp_extract(filename, '/r(\d+)-c(\d+)/', 1)::INTEGER AS repeat,
       regexp_extract(filename, '/r(\d+)-c(\d+)/', 2)::INTEGER AS concurrency,
       metadata.request_end_ns AS end_ns, metadata.request_start_ns AS start_ns
FROM read_json('results/qwen38-recipe-grid-20261019/*/artifacts/**/profile_export.jsonl',
               format='newline_delimited', filename=true,
               columns={metadata: 'STRUCT(request_end_ns BIGINT, request_start_ns BIGINT)'});

CREATE TEMP VIEW windows AS
SELECT CASE grid WHEN 'default' THEN 'Default' WHEN '2k' THEN 'Up to 2K' ELSE 'Up to 4K' END AS variant,
       grid, repeat, concurrency, make_timestamp_ns(list_sort(list(end_ns))[concurrency]) AS start_time,
       make_timestamp_ns(max(start_ns)) AS end_time
FROM requests GROUP BY grid, repeat, concurrency;

CREATE TEMP VIEW graph_lines AS
SELECT regexp_extract(filename, '/(default|2k|4k)/server.log$', 1) AS grid,
       unnest(string_split(content, chr(10))) AS line
FROM read_text('results/qwen38-recipe-grid-20261019/*/server.log');

CREATE TEMP VIEW graph_rows AS
SELECT grid,
       try_cast('2026-' || regexp_extract(line, 'INFO (\d\d-\d\d \d\d:\d\d:\d\d)', 1) AS TIMESTAMP) AS logged_at,
       (regexp_extract(line, '\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|', 1))::INTEGER AS tokens,
       regexp_extract(line, '\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|', 4) AS mode,
       (regexp_extract(line, '\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|', 5))::INTEGER AS steps
FROM graph_lines
WHERE regexp_matches(line, '\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\w+\s*\|\s*\d+\s*\|');

CREATE TEMP VIEW covered AS
SELECT w.variant, w.repeat, w.concurrency, g.* EXCLUDE(grid)
FROM windows w JOIN graph_rows g ON w.grid=g.grid
AND g.logged_at >= w.start_time + INTERVAL '10 seconds'
AND g.logged_at + INTERVAL '1 second' <= w.end_time;

COPY (SELECT * FROM covered) TO 'results/qwen38-recipe-grid-20261019/steady-graph-shapes.parquet' (FORMAT PARQUET);

SELECT variant, concurrency, count(DISTINCT (repeat, logged_at)) AS intervals,
       sum(steps) AS steps, sum(steps) FILTER (WHERE mode='NONE') AS eager_steps,
       sum(steps) FILTER (WHERE mode='PIECEWISE') AS piecewise_steps,
       sum(steps) FILTER (WHERE tokens>1024 AND tokens<=2048) AS tokens_1k_to_2k,
       sum(steps) FILTER (WHERE tokens>2048 AND tokens<=4096) AS tokens_2k_to_4k,
       sum(steps) FILTER (WHERE tokens>4096) AS tokens_above_4k
FROM covered WHERE concurrency IN (128,256) GROUP BY ALL ORDER BY concurrency,variant;

SELECT variant, concurrency, min(tokens) AS min_tokens, max(tokens) AS max_tokens,
       round(sum(tokens*steps)/sum(steps),1) AS mean_tokens, sum(steps) AS steps
FROM covered WHERE concurrency IN (128,256) AND tokens>1024
GROUP BY ALL ORDER BY concurrency,variant;

-- All concurrency levels: complete logging intervals within each client run.
-- Includes initial filling and draining, unlike the strict steady sample above.
CREATE TEMP VIEW full_windows AS
SELECT CASE grid WHEN 'default' THEN 'Default' WHEN '2k' THEN 'Up to 2K' ELSE 'Up to 4K' END AS variant,
       grid, repeat, concurrency, make_timestamp_ns(min(start_ns)) AS start_time,
       make_timestamp_ns(max(end_ns)) AS end_time
FROM requests GROUP BY grid, repeat, concurrency;

CREATE TEMP VIEW full_covered AS
SELECT w.variant, w.repeat, w.concurrency, g.* EXCLUDE(grid)
FROM full_windows w JOIN graph_rows g ON w.grid=g.grid
AND g.logged_at >= w.start_time + INTERVAL '10 seconds'
AND g.logged_at + INTERVAL '1 second' <= w.end_time;

CREATE TEMP VIEW run_coverage AS
SELECT variant, concurrency, repeat, count(DISTINCT logged_at) AS intervals, sum(steps) AS steps,
       sum(CASE WHEN mode='FULL' THEN steps ELSE 0 END) AS full_steps,
       sum(CASE WHEN mode='PIECEWISE' THEN steps ELSE 0 END) AS piecewise_steps,
       sum(CASE WHEN mode='NONE' THEN steps ELSE 0 END) AS eager_steps
FROM full_covered GROUP BY ALL;

CREATE TEMP VIEW coverage AS
SELECT variant, concurrency, count(*) AS runs, sum(intervals) AS intervals, sum(steps) AS steps,
       sum(full_steps) AS full_steps, sum(piecewise_steps) AS piecewise_steps, sum(eager_steps) AS eager_steps,
       100.0*(sum(full_steps)+sum(piecewise_steps))/sum(steps) AS graphed_pct
FROM run_coverage GROUP BY ALL;

COPY (SELECT * FROM run_coverage ORDER BY concurrency,variant,repeat)
TO 'results/qwen38-recipe-grid-20261019/graph-coverage-runs.csv' (HEADER);
COPY (SELECT * FROM coverage ORDER BY concurrency,variant)
TO 'results/qwen38-recipe-grid-20261019/graph-coverage.csv' (HEADER);

SELECT concurrency,
       round(max(graphed_pct) FILTER(WHERE variant='Default'),2) AS default_pct,
       round(max(graphed_pct) FILTER(WHERE variant='Up to 2K'),2) AS up_to_2k_pct,
       round(max(graphed_pct) FILTER(WHERE variant='Up to 4K'),2) AS up_to_4k_pct
FROM coverage GROUP BY concurrency ORDER BY concurrency;

SELECT CASE WHEN count(*)=54 AND count(DISTINCT (variant,concurrency,repeat))=54
       AND bool_and(steps=full_steps+piecewise_steps+eager_steps AND intervals>0)
       THEN true ELSE error('Incomplete or inconsistent coverage') END AS coverage_verified
FROM run_coverage;

-- Above the MTP3 decode-only bound, steps must contain prefill tokens.
CREATE TEMP VIEW prefill_run_coverage AS
SELECT variant, concurrency, repeat, sum(steps) AS steps,
       sum(CASE WHEN mode IN ('FULL','PIECEWISE') THEN steps ELSE 0 END) AS graphed_steps
FROM full_covered WHERE tokens>4*concurrency GROUP BY ALL;

COPY (SELECT variant, concurrency, count(*) AS runs, sum(steps) AS steps,
             sum(graphed_steps) AS graphed_steps, 100.0*sum(graphed_steps)/sum(steps) AS graphed_pct
      FROM prefill_run_coverage GROUP BY ALL ORDER BY concurrency,variant)
TO 'results/qwen38-recipe-grid-20261019/graph-coverage-prefill.csv' (HEADER);

SELECT CASE WHEN count(*)=54 AND bool_and(steps>0 AND graphed_steps<=steps)
       THEN true ELSE error('Incomplete prefill coverage') END AS prefill_coverage_verified
FROM prefill_run_coverage;
