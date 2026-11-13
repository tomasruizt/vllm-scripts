"""Check drafted >= verified >= accepted on a running vLLM server.

Sends one JSON-schema and one free-text request (temperature 0), then checks the
`detailed` per-request spec decode metrics at every step, and the Prometheus
counter deltas (including that the verified per-pos counters sum to the total).
"""

import argparse
import json
import re
import urllib.request

COUNTERS = ["num_drafts", "num_draft_tokens", "num_verified_tokens", "num_accepted_tokens"]
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "skills"],
}
PROMPTS = {
    "json": "Describe a fictional software engineer as JSON with name, age, skills.",
    "free": "Repeat the following list ten times: apple, banana, cherry, date.",
}


def main() -> None:
    args = parse_args()
    base = f"http://localhost:{args.port}"
    before = read_counters(base)
    rows = [check_request(base, args, kind) for kind in PROMPTS]
    delta = {k: v - before.get(k, 0) for k, v in read_counters(base).items()}
    check_prom(delta, args.expect_av)
    print(json.dumps({"config": args.name, "requests": rows, "prom_delta": delta}))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--expect-av", action="store_true")
    return p.parse_args()


def check_request(base: str, args: argparse.Namespace, kind: str) -> dict:
    body = {
        "model": get_model(base),
        "prompt": PROMPTS[kind],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    if kind == "json":
        body["structured_outputs"] = {"json": SCHEMA}
    sd = post(f"{base}/v1/completions", body)["metrics"]["speculative_decoding"]
    steps = zip(sd["per_step_drafted"], sd["per_step_verified"], sd["per_step_accepted"])
    for d, v, a in steps:
        assert d >= v >= a, (kind, d, v, a)
    totals = (sd["num_draft_tokens"], sd["num_verified_draft_tokens"], sd["num_accepted_draft_tokens"])
    assert totals[0] >= totals[1] >= totals[2], (kind, totals)
    assert sum(sd["per_step_verified"]) == totals[1], kind
    return {"kind": kind, "steps": sd["num_spec_steps"], "drafted": totals[0], "verified": totals[1], "accepted": totals[2]}


def check_prom(delta: dict[str, float], expect_av: bool) -> None:
    d, v, a = (delta[k] for k in COUNTERS[1:])
    assert d >= v >= a, (d, v, a)
    assert delta["num_verified_tokens_per_pos"] == v, delta
    if expect_av:
        assert v < d, (d, v)


def read_counters(base: str) -> dict[str, float]:
    text = urllib.request.urlopen(f"{base}/metrics").read().decode()
    names = COUNTERS + ["num_verified_tokens_per_pos"]
    out = dict.fromkeys(names, 0.0)
    for name in names:
        pat = rf"^vllm:spec_decode_{name}_total(?:{{[^}}]*}})? (\S+)$"
        out[name] = sum(float(x) for x in re.findall(pat, text, re.M))
    return out


def get_model(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models") as r:
        return json.load(r)["data"][0]["id"]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


if __name__ == "__main__":
    main()
