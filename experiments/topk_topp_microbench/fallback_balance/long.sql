-- raw.parquet (one row per case and pass) -> results.parquet (one row per case, pass, impl).
CREATE TABLE raw AS SELECT *, row_number() OVER () AS rid FROM 'raw.parquet';
CREATE TABLE t AS UNPIVOT (SELECT rid, COLUMNS('^.* \(ms\)$') FROM raw) ON COLUMNS('^.* \(ms\)$') INTO NAME k VALUE ms;
CREATE TABLE c AS UNPIVOT (SELECT rid, COLUMNS('^.* check$') FROM raw) ON COLUMNS('^.* check$') INTO NAME k VALUE exact;
COPY (
  SELECT split_part(ltrim(filename, './'), '/', 1) AS gpu, split_part(ltrim(filename, './'), '/', 2) AS suite,
         regexp_extract(filename, '_r(\d+)\.jsonl', 1)::INT AS pass,
         Source AS source, Layout AS layout, "Case" AS case_name, Batch AS batch, Vocab AS vocab,
         replace(t.k, ' (ms)', '') AS impl, 1000 * t.ms AS us, c.exact, "masks equal" AS masks_equal
  FROM raw JOIN t USING (rid) JOIN c ON c.rid = raw.rid AND replace(c.k, ' check', '') = replace(t.k, ' (ms)', '')
  WHERE t.ms IS NOT NULL
) TO 'results.parquet';
