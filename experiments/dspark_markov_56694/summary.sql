WITH grouped AS (
    SELECT mode, workload, concurrency, count(*) AS trials,
           median(throughput) AS throughput,
           min(throughput) AS throughput_min, max(throughput) AS throughput_max,
           median(mean_tpot_ms) AS tpot_ms,
           median(acceptance_length) AS al,
           median(aiperf_throughput) AS aiperf_throughput,
           sum(requests) AS requests
    FROM read_parquet('metrics.parquet')
    WHERE included
    GROUP BY ALL
)
SELECT p.workload, p.concurrency, p.trials,
       d.throughput AS dense_tok_s, b.throughput AS base32_tok_s,
       p.throughput AS union32_tok_s,
       100 * (p.throughput / d.throughput - 1) AS union_vs_dense_pct,
       100 * (p.throughput / b.throughput - 1) AS union_vs_base_pct,
       d.tpot_ms AS dense_tpot_ms, p.tpot_ms AS union_tpot_ms,
       100 * (p.tpot_ms / d.tpot_ms - 1) AS union_tpot_change_pct,
       d.al AS dense_al, b.al AS base32_al, p.al AS union_al,
       d.throughput_min AS dense_min, d.throughput_max AS dense_max,
       p.throughput_min AS union_min, p.throughput_max AS union_max,
       100 * (p.aiperf_throughput / d.aiperf_throughput - 1) AS aiperf_gain_pct,
       d.requests AS dense_requests, p.requests AS union_requests
FROM grouped p
JOIN grouped d USING (workload, concurrency)
LEFT JOIN grouped b ON b.workload=p.workload AND b.concurrency=p.concurrency AND b.mode='base32'
WHERE p.mode='union32' AND d.mode='dense'
ORDER BY length(p.workload), p.workload, p.concurrency;
