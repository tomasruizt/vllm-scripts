"""Combine standalone reports into offline model tabs using their existing theme."""

import html
import re


def model_tabs(reports, *, title):
    styles, selectors, choices, labels, panels = [], [], [], [], []
    for index, (slug, label, page) in enumerate(reports):
        ids = set(re.findall(r'\bid="([^"]+)"', page))
        if ids:
            pattern = (
                r"(?<![\w-])("
                + "|".join(map(re.escape, sorted(ids, key=len, reverse=True)))
                + r")(?![\w-])"
            )
            page = re.sub(pattern, lambda match, slug=slug: f"{slug}-{match[0]}", page)
        page = page.replace('name="table"', f'name="{slug}-table"')
        styles.extend(re.findall(r"<style>(.*?)</style>", page, flags=re.DOTALL))
        body = re.search(r"<body>(.*?)</body>", page, flags=re.DOTALL)[1]
        checked = " checked" if index == 0 else ""
        choices.append(
            f'<input class="metric-choice" type="radio" name="model" id="model-{slug}" aria-controls="panel-{slug}"{checked}>'
        )
        labels.append(f'<label for="model-{slug}">{html.escape(label)}</label>')
        panels.append(
            f'<section class="model-panel" id="panel-{slug}">{body}</section>'
        )
        selectors.append(
            f"#model-{slug}:checked~.model-panels>#panel-{slug}{{display:block}}"
        )
        selectors.append(
            f'#model-{slug}:checked~.model-tabs label[for="model-{slug}"]{{background:#235cca;color:white}}'
        )
        selectors.append(
            f'#model-{slug}:focus-visible~.model-tabs label[for="model-{slug}"]{{outline:3px solid #e4ad31;outline-offset:3px}}'
        )
    css = "\n".join(styles + [".model-panel{display:none}"] + selectors)
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{css}</style></head><body>"
        + "\n".join(choices)
        + '<div class="metric-tabs model-tabs">'
        + "".join(labels)
        + "</div>"
        + '<div class="model-panels">'
        + "".join(panels)
        + "</div></body></html>"
    )
