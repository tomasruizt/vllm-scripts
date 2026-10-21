-- Run from the experiment directory after make analyze-av-full-sweep.
CREATE MACRO counter_total(blob, name) AS (
    SELECT sum(json_extract(value, '$.stats.total')::DOUBLE)
    FROM json_each(json_extract(blob, '$.metrics."' || name || '".series'))
);

CREATE TABLE runs AS
SELECT mode, stage, trial, concurrency, artifact, throughput, acceptance_length,
    counter_total(aiperf_server_metrics_json, 'vllm:spec_decode_num_drafts') attempts,
    counter_total(aiperf_server_metrics_json, 'vllm:spec_decode_num_draft_tokens') proposed,
    counter_total(aiperf_server_metrics_json, 'vllm:spec_decode_num_accepted_tokens') accepted,
    aiperf_server_metrics_json
FROM 'results/qwen-av-full-sweep-20261020/analysis/metrics.parquet' WHERE included;

CREATE TABLE positions AS
SELECT r.artifact, r.mode, r.concurrency, json_extract_string(s.value, '$.labels.position')::INT pos,
    json_extract(s.value, '$.stats.total')::DOUBLE accepted,
    json_extract(s.value, '$.stats.total')::DOUBLE / r.attempts survival
FROM runs r, json_each(json_extract(r.aiperf_server_metrics_json,
    '$.metrics."vllm:spec_decode_num_accepted_tokens_per_pos".series')) s;

COPY (SELECT * EXCLUDE(aiperf_server_metrics_json), proposed/attempts proposed_per_attempt,
    1+accepted/attempts recomputed_al FROM runs)
TO 'results/qwen-av-full-sweep-20261020/analysis/acceptance-runs.parquet' (FORMAT PARQUET);
COPY positions TO 'results/qwen-av-full-sweep-20261020/analysis/acceptance-positions.parquet' (FORMAT PARQUET);

COPY (
    WITH p AS (SELECT artifact, sum(accepted) accepted, sum(survival) survival FROM positions GROUP BY artifact),
    monotone AS (SELECT *,lag(accepted) OVER(PARTITION BY artifact ORDER BY pos) previous FROM positions)
    SELECT count(*) runs,
        count(*) FILTER(WHERE abs(r.acceptance_length-(1+r.accepted/r.attempts))>1e-10) al_mismatches,
        count(*) FILTER(WHERE r.proposed<>7*r.attempts) proposed_count_mismatches,
        count(*) FILTER(WHERE p.accepted<>r.accepted) position_sum_mismatches,
        (SELECT count(*) FROM monotone WHERE accepted>previous OR survival<0 OR survival>1) invalid_survival_points
    FROM runs r JOIN p USING(artifact)
) TO 'results/qwen-av-full-sweep-20261020/analysis/acceptance-validation.csv' (HEADER);

COPY (
    WITH g AS (SELECT mode,concurrency,median(acceptance_length) al, median(throughput) throughput,
        min(acceptance_length) al_min,max(acceptance_length) al_max FROM runs GROUP BY mode,concurrency)
    SELECT n.concurrency,n.al native_al,m.al measured_al,m.al-n.al al_delta,
        100*(m.al/n.al-1) al_change_pct,100*(m.throughput/n.throughput-1) throughput_change_pct,
        n.al_min native_al_min,n.al_max native_al_max,m.al_min measured_al_min,m.al_max measured_al_max
    FROM g n JOIN g m USING(concurrency) WHERE n.mode='native' AND m.mode='measured' ORDER BY concurrency
) TO 'results/qwen-av-full-sweep-20261020/analysis/acceptance-comparison.csv' (HEADER);

SELECT * FROM read_csv_auto('results/qwen-av-full-sweep-20261020/analysis/acceptance-validation.csv');
SELECT concurrency,round(native_al,3) native_al,round(measured_al,3) measured_al,
    round(al_change_pct,2) al_change_pct,round(throughput_change_pct,2) throughput_change_pct
FROM read_csv_auto('results/qwen-av-full-sweep-20261020/analysis/acceptance-comparison.csv');
SELECT mode,concurrency,list(round(survival,3) ORDER BY pos) survival
FROM (SELECT mode,concurrency,pos,median(survival) survival FROM positions GROUP BY ALL)
GROUP BY mode,concurrency ORDER BY concurrency,mode;
