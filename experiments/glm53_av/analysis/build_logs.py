"""Archive benchmark sessions and build a browsable log index."""

import argparse
import hashlib
import html
import json
import shutil
import tarfile
from pathlib import Path

CODE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CODE.parent / "report")
    parser.add_argument(
        "--sources", type=Path, help="JSON mapping session names to directories"
    )
    args = parser.parse_args()
    build_logs(args.output, args.sources)


def build_logs(output, sources=None):
    """Refresh the log index, checksums, and archive from saved sessions."""
    output = output.resolve()
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if sources:
        for name, directory in json.loads(sources.read_text()).items():
            source = Path(directory)
            result = json.loads((source / "result.json").read_text())
            if result["status"] != "completed":
                raise ValueError(f"Incomplete benchmark session: {name}")
            if not (source / "server.log").is_file():
                raise FileNotFoundError(source / "server.log")
            destination = logs / name
            destination.mkdir(parents=True, exist_ok=True)
            for path in source.iterdir():
                if path.is_file() and path.suffix in (
                    ".log",
                    ".json",
                    ".txt",
                    ".py",
                    ".sh",
                ):
                    shutil.copy2(path, destination / path.name)
    sessions = sorted(path for path in logs.iterdir() if path.is_dir())
    rows = []
    for session in sessions:
        result = json.loads((session / "result.json").read_text())
        runs = []
        for path in sorted(session.glob("gsm8k-*-client.log")):
            label = path.name.removeprefix("gsm8k-").removesuffix("-client.log")
            runs.append(link(session, path.name, label))
        cache = "off" if result.get("prefix_caching") is False else "on"
        notes = "Supplementary; not plotted" if cache == "off" else "Plotted runs"
        cells = [
            html.escape(session.name),
            cache,
            notes,
            link(session, "server.log", "Server"),
            ", ".join(runs),
            link(session, "result.json", "Results")
            + " · "
            + link(session, "run.py", "Launch script"),
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    manifest = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for session in sessions
        for path in sorted(session.iterdir())
        if path.is_file()
    }
    (logs / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    theme = (CODE.parents[2] / "reporting/b200-theme.css").read_text()
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>GLM-5.3 DEP4 logs</title>
<style>{theme}
td {{ white-space: normal; text-align: left; }} th {{ text-align: left; }} td a {{ white-space: nowrap; }}
</style></head><body><h1>GLM-5.3 DEP4 logs</h1>
<p><a href="index.html">Report</a> · <a href="benchmark-logs.tar.gz" download>Download all logs and raw results</a> · <a href="logs/sha256.json">SHA256 checksums</a></p>
<section class="card"><div class="table-scroll"><table><thead><tr><th>Session</th><th>Prefix cache</th><th>Scope</th><th>Server</th><th>Benchmarks</th><th>Reproduce</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></section>
<p>The download includes server and client logs, warmups, launch scripts, results, and before/after metric snapshots. Sessions sharing a server list each benchmark separately.</p></body></html>"""
    (output / "benchmark-logs.html").write_text(page)
    with tarfile.open(output / "benchmark-logs.tar.gz", "w:gz") as archive:
        archive.add(logs, arcname="logs")
    print(f"Archived {len(sessions)} sessions and {len(manifest)} files")


def link(session, filename, label):
    path = session / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return f'<a href="logs/{html.escape(session.name)}/{html.escape(filename)}">{html.escape(label)}</a>'


if __name__ == "__main__":
    main()
