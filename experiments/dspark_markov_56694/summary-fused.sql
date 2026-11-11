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
SELECT u.workload, u.concurrency, u.trials,
       d.throughput AS dense_tok_s, u.throughput AS union_tok_s, f.throughput AS unfused_tok_s,
       100 * (u.throughput / d.throughput - 1) AS union_vs_dense_pct,
       100 * (f.throughput / d.throughput - 1) AS unfused_vs_dense_pct,
       100 * (u.throughput / f.throughput - 1) AS fusion_pct,
       d.tpot_ms AS dense_tpot_ms, u.tpot_ms AS union_tpot_ms, f.tpot_ms AS unfused_tpot_ms,
       100 * (u.tpot_ms / d.tpot_ms - 1) AS union_tpot_change_pct,
       d.al AS dense_al, u.al AS union_al, f.al AS unfused_al,
       100 * (u.aiperf_throughput / d.aiperf_throughput - 1) AS union_whole_run_pct,
       100 * (f.aiperf_throughput / d.aiperf_throughput - 1) AS unfused_whole_run_pct,
       d.throughput_min AS dense_min, d.throughput_max AS dense_max,
       u.throughput_min AS union_min, u.throughput_max AS union_max,
       u.requests AS union_requests
FROM grouped u
JOIN grouped d ON d.workload = u.workload AND d.concurrency = u.concurrency AND d.mode = 'dense'
JOIN grouped f ON f.workload = u.workload AND f.concurrency = u.concurrency AND f.mode = 'union32_unfused'
WHERE u.mode = 'union32'
ORDER BY length(u.workload), u.workload, u.concurrency;
