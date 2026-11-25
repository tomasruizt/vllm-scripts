# One row per human prompt, assistant text, tool call and tool result of a session transcript.
def row($r; $kind; $name; $err; $text; $id):
  {sid: $r.sessionId, ts: $r.timestamp, kind: $kind, name: $name, err: $err, text: $text, id: $id,
   src: ($r.promptSource // null)};

select(.type == "user" or .type == "assistant")
| select(.timestamp >= $since and .timestamp < $until)
| . as $r | $r.message.content as $c
| if ($c | type) == "string" then row($r; "user_text"; null; null; $c[0:3000]; null)
  else $c[]?
    | if .type == "text" then row($r; $r.type + "_text"; null; null; .text[0:3000]; null)
      elif .type == "tool_use" then row($r; "tool_use"; .name; null; (.input | tojson | .[0:2000]); .id)
      elif .type == "tool_result" then
        row($r; "tool_result"; null; (.is_error // false);
            ((.content | if type == "string" then . else (map(.text? // "") | join("\n")) end)[0:1500]);
            .tool_use_id)
      else empty end
  end
