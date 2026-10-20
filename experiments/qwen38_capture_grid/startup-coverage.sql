SET memory_limit='4GB';
SET threads=2;
CREATE TEMP VIEW requests AS
SELECT coalesce(nullif(regexp_extract(filename, 'startup-(\d+)-', 1), '')::INTEGER, 1) AS startup,
       regexp_extract(filename, '/(default|2k|4k)/artifacts/', 1) AS variant,
       regexp_extract(filename, '/r(\d+)-c(\d+)/', 1)::INTEGER AS repeat,
       regexp_extract(filename, '/r(\d+)-c(\d+)/', 2)::INTEGER AS concurrency,
       metadata.request_start_ns AS start_ns, metadata.request_end_ns AS end_ns
FROM read_json(['results/qwen38-recipe-grid-20261019/*/artifacts/**/profile_export.jsonl',
                'results/qwen38-recipe-restarts-20261020/qwen38-recipe-*/*/artifacts/**/profile_export.jsonl'],
               format='newline_delimited', filename=true,
               columns={metadata: 'STRUCT(request_end_ns BIGINT, request_start_ns BIGINT)'});

CREATE TEMP VIEW windows AS
SELECT startup,variant,repeat,concurrency,make_timestamp_ns(min(start_ns)) AS start_time,
       make_timestamp_ns(max(end_ns)) AS end_time
FROM requests GROUP BY ALL;

CREATE TEMP VIEW lines AS
SELECT coalesce(nullif(regexp_extract(filename, 'startup-(\d+)-', 1), '')::INTEGER, 1) AS startup,
       regexp_extract(filename, '/(default|2k|4k)/server.log$', 1) AS variant,
       unnest(string_split(content,chr(10))) AS line
FROM read_text(['results/qwen38-recipe-grid-20261019/*/server.log',
                'results/qwen38-recipe-restarts-20261020/qwen38-recipe-*/*/server.log']);

CREATE TEMP VIEW shapes AS
SELECT startup,variant,
       try_cast('2026-' || regexp_extract(line,'INFO (\d\d-\d\d \d\d:\d\d:\d\d)',1) AS TIMESTAMP) AS logged_at,
       regexp_extract(line,'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|',1)::INTEGER AS tokens,
       regexp_extract(line,'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|',4) AS mode,
       regexp_extract(line,'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|',5)::INTEGER AS steps
FROM lines WHERE regexp_matches(line,'\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\w+\s*\|\s*\d+\s*\|');

CREATE TEMP VIEW covered AS
SELECT w.startup,w.variant,w.repeat,w.concurrency,s.logged_at,s.tokens,s.mode,s.steps
FROM windows w JOIN shapes s USING(startup,variant)
WHERE s.logged_at>=w.start_time+INTERVAL '10 seconds' AND s.logged_at+INTERVAL '1 second'<=w.end_time;

COPY (SELECT * FROM covered) TO 'results/qwen38-recipe-restarts-20261020/graph-shapes.parquet' (FORMAT PARQUET);

CREATE TEMP VIEW runs AS
SELECT startup,variant,repeat,concurrency,sum(steps) AS steps,
       sum(CASE WHEN mode IN ('FULL','PIECEWISE') THEN steps ELSE 0 END) AS graphed_steps
FROM covered GROUP BY ALL;
SELECT CASE WHEN count(*)=90 AND bool_and(steps>0) THEN true ELSE error('Incomplete graph coverage') END AS verified FROM runs;

COPY (SELECT * FROM runs ORDER BY startup,variant,repeat,concurrency)
TO 'results/qwen38-recipe-restarts-20261020/graph-coverage-runs.csv' (HEADER);
COPY (SELECT CASE variant WHEN 'default' THEN 'Default' WHEN '2k' THEN 'Up to 2K' ELSE 'Up to 4K' END AS variant,
             concurrency,count(*) AS runs,sum(steps) AS steps,sum(graphed_steps) AS graphed_steps,
             100.0*sum(graphed_steps)/sum(steps) AS graphed_pct
      FROM runs GROUP BY ALL ORDER BY concurrency,variant)
TO 'results/qwen38-recipe-restarts-20261020/graph-coverage.csv' (HEADER);
CREATE TEMP VIEW prefill_runs AS
SELECT startup,variant,repeat,concurrency,sum(steps) AS steps,
       sum(CASE WHEN mode IN ('FULL','PIECEWISE') THEN steps ELSE 0 END) AS graphed_steps,
       100.0*graphed_steps/sum(steps) AS graphed_pct
FROM covered WHERE tokens>4*concurrency GROUP BY ALL;
SELECT CASE WHEN count(*)=90 AND bool_and(steps>0) THEN true ELSE error('Incomplete prefill coverage') END AS verified FROM prefill_runs;

COPY (SELECT CASE variant WHEN 'default' THEN 'Default' WHEN '2k' THEN 'Up to 2K' ELSE 'Up to 4K' END AS variant,
             concurrency,count(*) AS runs,sum(steps) AS steps,sum(graphed_steps) AS graphed_steps,
             100.0*sum(graphed_steps)/sum(steps) AS graphed_pct,
             avg(graphed_pct) AS mean_pct,min(graphed_pct) AS min_pct,max(graphed_pct) AS max_pct
      FROM prefill_runs GROUP BY ALL ORDER BY concurrency,variant)
TO 'results/qwen38-recipe-restarts-20261020/graph-coverage-prefill.csv' (HEADER);
