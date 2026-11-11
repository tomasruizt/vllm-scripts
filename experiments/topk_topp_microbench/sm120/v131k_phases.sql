-- Phase table of a phases_v131k.py run: duckdb -c ".read sm120/v131k_phases.sql" with FILE set by sed, e.g.
-- sed s/FILE/results\\/sm120-v131k\\/rtx_base.jsonl/ sm120/v131k_phases.sql | duckdb
with t as (select * from read_json_auto('FILE', union_by_name = true)),
ph as (select * from t where impl = 'new_phases')
select ph.scenario s, ph.batch b, ph.vocab // 1000 || 'k' v,
    (select us from t where impl = 'triton' and t.scenario = ph.scenario and t.batch = ph.batch and t.vocab = ph.vocab) main,
    (select us from t where impl = 'new' and t.scenario = ph.scenario and t.batch = ph.batch and t.vocab = ph.vocab) pr,
    ph.levels, ph."select" sel, ph.r0_hist hist, ph.r0_plan plan, ph.r0_gather gather, ph.r0_cut cut, ph.fill,
    round(ph.gb / ph.levels * 1e3, 2) lv_tbs, round(ph.gb / ph.r0_hist * 1e3, 2) hist_tbs,
    round(ph.gb / ph.r0_gather * 1e3, 2) gather_read_tbs
from ph order by s, ph.vocab, b;
