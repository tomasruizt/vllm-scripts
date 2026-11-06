#!/usr/bin/env bash
# Sample host CPU load every INTERVAL seconds into OUT until killed. One line per sample:
#   <UTC ISO time> <5 /proc/loadavg fields> <busy cores over the interval, from /proc/stat>
# Usage: host_load.sh OUT [INTERVAL]   Start it in the background next to the server, kill it on cleanup.
set -u
out=$1 interval=${2:-5}
ncpu=$(nproc)

cpu_times() {  # total and idle jiffies of all CPUs
    local _ user nice system idle iowait irq softirq steal rest
    read -r _ user nice system idle iowait irq softirq steal rest < /proc/stat
    echo "$((user + nice + system + idle + iowait + irq + softirq + steal)) $((idle + iowait))"
}

read -r prev_total prev_idle < <(cpu_times)
while sleep "$interval"; do
    read -r total idle < <(cpu_times)
    busy=$(awk -v dt=$((total - prev_total)) -v di=$((idle - prev_idle)) -v n="$ncpu" \
        'BEGIN { printf "%.1f", (dt > 0 ? (dt - di) / dt * n : 0) }')
    echo "$(date -u +%FT%TZ) $(cat /proc/loadavg) $busy"
    prev_total=$total prev_idle=$idle
done > "$out"
