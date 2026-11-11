"""Replace the microbenchmark tables of the #66792 description with pr_tables_queue.py output.

Usage: apply_tables.py TABLES.md BODY.md   (rewrites BODY.md in place)
"""

import re
import sys

HEADINGS = {"b300": "B300", "h100": "H100", "a100": "A100", "rtx": "RTX PRO 6000", "mi355x": "MI355X"}
ANCHORS = {
    "qwen-topk-topp": "{g}, real Qwen3.6 rows, top-k 20 and top-p 0.95",
    "glm-topp": "{g}, real GLM-5.3-Flash verify rows, top-p 0.95",
    "qwen-topp": "{g}, real Qwen3.6 rows, top-p 0.95 only",
    "synthetic": "{g}, synthetic Zipf scenarios",
}


def main() -> None:
    tables = dict(re.findall(r"^### (\S+)\n((?:\|.*\n?)+)", open(sys.argv[1]).read(), re.M))
    body = open(sys.argv[2]).read()
    body = replace_claims(body, tables["claims"].strip().splitlines())
    for key, table in tables.items():
        if key == "claims":
            continue
        gpu, name = key.split("/")
        if gpu not in HEADINGS:
            continue
        anchor = ANCHORS[name].format(g=HEADINGS[gpu])
        start = body.find(anchor)
        if start < 0:
            continue
        a = body.index("\n|", start) + 1
        b = re.compile(r"\n(?!\|)").search(body, a).start() + 1
        body = body[:a] + table.strip() + "\n" + body[b:]
    open(sys.argv[2], "w").write(body)


def replace_claims(body: str, rows: list[str]) -> str:
    for row in rows:
        name = row.split("|")[1].strip()
        pattern = re.compile(rf"^(\s*)\| {re.escape(name)} \|.*$", re.M)
        body = pattern.sub(lambda m: m.group(1) + row + ("[^l2]" if False else ""), body, count=1)
    return body


if __name__ == "__main__":
    main()
