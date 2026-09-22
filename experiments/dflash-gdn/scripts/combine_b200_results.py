"""Combine both draft-length reports with native, JavaScript-free tabs."""

import argparse
import html
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit

from render_b200_results import MODELS, tab_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block8", type=Path, required=True)
    parser.add_argument("--block16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    controls, tabs, panels, styles = [], [], [], []
    model_controls, model_tabs = [], []
    for index, model in enumerate(MODELS):
        key = f"shared-model-{model.lower()}"
        control, tab, rules = tab_control(key, model, "shared-model", index == 0)
        targets = " ".join(f"draft-{k}-panel-model-{model.lower()}" for k in (7, 15))
        model_controls.append(control.replace(f'aria-controls="panel-{key}"', f'aria-controls="{targets}"'))
        model_tabs.append(tab)
        styles.extend(rules[1:])
        for k in (7, 15):
            styles.append(f'#{key}:checked ~ .panels #draft-{k}-panel-model-{model.lower()} {{ display: block; }}')
    for index, (root, proposals, label) in enumerate([
        (args.block8, 7, "K=7/8"), (args.block16, 15, "K=15/16")
    ]):
        subprocess.run([sys.executable, str(Path(__file__).with_name("render_b200_results.py")),
                        str(root), "--num-speculative-tokens", str(proposals)], check=True)
        source = (root / "RESULTS.html").read_text()
        key = f"draft-{proposals}"
        source = namespace_ids(source, key)
        source = relocate_links(source, root, args.output.parent)
        styles.append(re.search(r"<style>(.*?)</style>", source, re.S)[1])
        body = re.search(r"(<main>.*?</footer>)", source, re.S)[1]
        body = re.sub(rf'<input\b[^>]*\bname="{key}-model"[^>]*>', '', body)
        body = re.sub(r'<div class="metric-tabs model-tabs">.*?</div>', '', body, flags=re.S)
        for model in MODELS:
            body = body.replace(f'aria-labelledby="{key}-label-model-{model.lower()}"',
                                f'aria-labelledby="label-shared-model-{model.lower()}"')
        body = body.replace("<main>", '<div class="draft-report">').replace("</main>", "</div>")
        body = body.replace(
            "Share the HTML, benchmark-logs.html, and model directories together.",
            "Share this HTML with both result directories intact.",
        )
        control, tab, rules = tab_control(key, label, "draft-length", index == 0)
        controls.append(control)
        tabs.append(tab)
        styles.extend(rules)
        panels.append(f'<section class="tab-panel" id="panel-{key}" aria-labelledby="label-{key}">{body}</section>')
    document = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>B200 benchmarks — draft-length comparison</title><style>',
        *styles, '</style></head><body><h1>B200 benchmark results</h1>',
        '<p class="subtitle">vLLM 0.30.0 · SGLang 0.5.20 · PR 52297 · 180/180 points</p>',
        '<p>vLLM counts only proposed tokens; SGLang counts the full block, including one conditioning token. Thus vLLM K=7 equals SGLang K=8, and vLLM K=15 equals SGLang K=16.</p>',
        '<main role="group" aria-label="Benchmark configuration">', *controls, *model_controls,
        '<div class="metric-tabs" role="group" aria-label="Draft length">', *tabs, '</div>',
        '<div class="metric-tabs model-tabs" role="group" aria-label="Model size">',
        *model_tabs, '</div><div class="panels">',
        *panels, '</div></main></body></html>',
    ]
    args.output.write_text('\n'.join(document))
    print(f"Wrote {args.output}")


def namespace_ids(source, prefix):
    ids = {value: f"{prefix}-{value}" for value in re.findall(r'\bid="([^"]+)"', source)}
    source = re.sub(r'\b(id|for|aria-controls|aria-labelledby)="([^"]+)"',
                    lambda m: f'{m[1]}="{ids.get(m[2], m[2])}"', source)
    source = re.sub(r'#([\w.-]+)', lambda m: '#' + ids.get(m[1], m[1]), source)
    return re.sub(r'\bname="([^"]+)"', lambda m: f'name="{prefix}-{m[1]}"', source)


def relocate_links(source, root, output_dir):
    prefix = Path(os.path.relpath(root.resolve(), output_dir.resolve())).as_posix()

    def replace(match):
        href = html.unescape(match[1])
        if href.startswith('#') or urlsplit(href).scheme or href.startswith('//'):
            return match[0]
        return f'href="{html.escape(quote(prefix, safe="/.") + "/" + href, quote=True)}"'

    return re.sub(r'\bhref="([^"]+)"', replace, source)


if __name__ == "__main__":
    main()
