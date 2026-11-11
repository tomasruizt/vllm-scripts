.bail on

CREATE TEMP VIEW cells AS
SELECT * FROM read_parquet(getvariable('serving_dir') || '/serving.parquet');
CREATE TEMP VIEW requests AS
SELECT * FROM read_parquet(getvariable('serving_dir') || '/serving-requests.parquet');

COPY (
    SELECT n.repeat, n.concurrency, n.throughput AS native_throughput, f.throughput AS candidate_throughput,
           100 * (f.throughput / n.throughput - 1) AS throughput_gain_pct,
           n.mean_tpot_ms AS native_tpot_ms, f.mean_tpot_ms AS candidate_tpot_ms,
           100 * (1 - f.mean_tpot_ms / n.mean_tpot_ms) AS tpot_reduction_pct,
           n.acceptance_length AS native_al, f.acceptance_length AS candidate_al
    FROM cells n JOIN cells f USING (repeat, concurrency)
    WHERE n.mode='native' AND f.mode='flashinfer-deterministic' AND n.valid AND f.valid
) TO (getvariable('serving_dir') || '/serving-pairs.parquet') (FORMAT PARQUET);

COPY (
    SELECT n.repeat, n.concurrency, count(*) AS shared_requests,
           count(*) FILTER (WHERE n.output_tokens=f.output_tokens) AS same_lengths,
           avg(abs(n.output_tokens-f.output_tokens)) AS mean_length_difference,
           avg(n.tpot_ms) AS native_tpot_ms, avg(f.tpot_ms) AS candidate_tpot_ms,
           100 * (1 - avg(f.tpot_ms) / avg(n.tpot_ms)) AS tpot_reduction_pct
    FROM requests n JOIN requests f USING (repeat, concurrency, conversation_id, turn_index)
    WHERE n.mode='native' AND f.mode='flashinfer-deterministic'
    GROUP BY ALL
) TO (getvariable('serving_dir') || '/shared-prompt-pairs.parquet') (FORMAT PARQUET);

COPY (
    SELECT a.mode, a.repeat AS repeat_a, b.repeat AS repeat_b, a.concurrency, count(*) AS shared_requests,
           count(*) FILTER (WHERE a.output_tokens=b.output_tokens) AS same_lengths,
           avg(abs(a.output_tokens-b.output_tokens)) AS mean_length_difference
    FROM requests a JOIN requests b USING (mode, concurrency, conversation_id, turn_index)
    WHERE b.repeat=a.repeat+1
    GROUP BY ALL
) TO (getvariable('serving_dir') || '/repeat-controls.parquet') (FORMAT PARQUET);

SELECT concurrency, median(throughput_gain_pct) AS median_throughput_gain_pct,
       min(throughput_gain_pct) AS min_gain_pct, max(throughput_gain_pct) AS max_gain_pct,
       median(tpot_reduction_pct) AS median_tpot_reduction_pct
FROM read_parquet(getvariable('serving_dir') || '/serving-pairs.parquet') GROUP BY ALL ORDER BY concurrency;
