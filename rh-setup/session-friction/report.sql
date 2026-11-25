create or replace table ev as select * from read_json_auto('ev.jsonl', maximum_object_size=50000000);

-- Human prompts only: 'sdk' is the IDE extension, 'typed' is the CLI.
create or replace view prompts as
select sid, ts, regexp_replace(regexp_replace(text, '<ide_[a-z_]+>.*?</ide_[a-z_]+>', '', 'gs'), '\s+', ' ', 'g') t
from ev where kind = 'user_text' and src in ('sdk', 'typed') and length(trim(text)) > 0;

select count(distinct sid) sessions, count(*) prompts, min(ts)::date first_day, max(ts)::date last_day from prompts;

-- Keyword categories catch only part of the real friction. Compare rates between windows, and read prompts.txt.
with c as (
  select lower(t) t from prompts
), cats as (
  select 'status_poll' cat, regexp_matches(t, 'status|did you find|is .* running|finished') hit from c
  union all select 'show_me_link', regexp_matches(t, 'show me .*(link|plot|file|table)|show me the pr') from c
  union all select 'too_verbose', regexp_matches(t, 'concise|shorter|too long|too much|cut .* by|simplify|not reading') from c
  union all select 'commit_push', regexp_matches(t, '^(commit|push|amend)|commit and push|push now|should commit|did you commit') from c
  union all select 'worktree', regexp_matches(t, 'worktree|close the local|remove the tree') from c
  union all select 'verify', regexp_matches(t, 'fact.?check|are you sure|did you read|still correct|check this|dive deep') from c
  union all select 'overbuilt', regexp_matches(t, 'hack|complicated|unnecessary|not the solution|dont need|don.t need|no need|too constrained|placeholders') from c
  union all select 'frustration', regexp_matches(t, 'idiot|stupid|wtf|fuck|slop|\?\?\?|bro |brother') from c
)
select cat, count(*) filter (where hit) n, round(100.0 * count(*) filter (where hit) / count(*), 1) per_100_prompts
from cats group by cat order by n desc;

select u.name tool, count(*) calls, count(*) filter (where r.err) errors
from ev u left join (select id, err from ev where kind = 'tool_result') r using (id)
where u.kind = 'tool_use' group by 1 order by calls desc limit 8;

copy (select left(sid::varchar, 4) || ' ' || strftime(ts, '%d %H:%M') || ' | ' || left(t, 330) from prompts order by sid, ts)
  to 'prompts.txt' (header false, quote '', delimiter E'\t');
copy (select left(u.sid::varchar, 4) sid, left(regexp_replace(u.text, '\s+', ' ', 'g'), 230) tool_call, left(regexp_replace(r.text, '\s+', ' ', 'g'), 260) result
      from ev u join (select id, err, text from ev where kind = 'tool_result') r using (id)
      where u.kind = 'tool_use' and r.err order by u.sid, u.ts)
  to 'errors.txt' (header true, delimiter E'\t');
