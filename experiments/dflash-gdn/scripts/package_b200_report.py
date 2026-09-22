"""Package the combined report for offline sharing or GitHub Pages."""

import argparse
import hashlib
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "dist/b200-report")
    args = parser.parse_args()
    source = args.results.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Output already exists; choose a fresh directory: {output}")
    files = collect_links(source / "RESULTS.html", source)
    for root_name in ("b200-block8-20260922", "b200-latest-20260922"):
        root = source / root_name
        for pattern in ("*/*/*/c*/summary.json", "*/*/*/c*/run_config.json", "plots/*"):
            files.update(path for path in root.glob(pattern) if path.is_file() and path.suffix != ".pdf")
        if (root / "experiment.json").exists():
            files.add(root / "experiment.json")
    for path in sorted(files):
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(output / "RESULTS.html", output / "index.html")
    (output / ".nojekyll").touch()
    (output / "README.md").write_text(
        "# B200 DFlash benchmark report\n\n"
        "- Extract the entire ZIP and open [index.html](index.html). Keep the directory structure intact so Server and Bench links work.\n"
        "- Includes both K settings, all three models, server/benchmark logs, original AIPerf metric and server-metric JSON exports, per-point summaries and configurations, plots, environment versions, and validation.\n"
        "- Follow Bench for each concurrency's source files and the metric field/formula mapping. Aggregates are included; independently recomputing percentiles requires the excluded per-request traces.\n"
        "- Model selection stays constant when switching K; no JavaScript or internet connection is required to view the report.\n"
        "- Excludes raw profiler traces and per-request datasets; this is a report bundle, not the full experiment archive.\n\n"
        "## Share on GitHub\n\n"
        "- **Quickest:** attach the ZIP to a GitHub release and share the download link. Readers extract it and open index.html.\n"
        "- **Best browsing experience:** publish this folder with GitHub Pages, then share the resulting website URL. GitHub repository file previews do not display the interactive HTML report.\n"
        "- For a dedicated report repository: copy this folder's contents into docs/, commit and push, then select Settings → Pages → Deploy from a branch → main → /docs. Keep .nojekyll.\n"
        "- For an existing Pages site, put this folder in a subdirectory of its publishing source.\n\n"
        "[GitHub Pages setup](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)\n"
    )
    collect_links(output / "index.html", output)
    manifest = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output.with_suffix(".zip")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent))
    print(f"Site: {output / 'index.html'}")
    print(f"ZIP: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")
    print(f"Validated local HTML links; packaged {len(manifest)} files plus checksum manifest.")


def collect_links(entry, root):
    pending, visited = [entry], set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or out-of-root report link: {path}")
        visited.add(path)
        if path.suffix != ".html":
            continue
        parser = Links()
        parser.feed(path.read_text())
        for link in parser.links:
            parsed = urlsplit(link)
            if not parsed.scheme and not parsed.netloc and parsed.path:
                pending.append(path.parent / unquote(parsed.path))
    return visited


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("href", "src") and value)


if __name__ == "__main__":
    main()
