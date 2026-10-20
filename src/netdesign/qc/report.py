"""Assemble ``qc/*.json`` and figures into a self-contained ``qc/report.html``."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from netdesign.qc.style import STATUS, STATUS_ICON

__all__ = ["build_report"]

STAGES = [
    ("inputs", "Inputs"),
    ("ensemble", "Velocity ensemble"),
    ("sensitivity", "Travel times and predicted location quality"),
    ("validation", "NonLinLoc validation"),
    ("design", "Network design"),
]

CSS = """
:root { --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --rule:#e3e2de; }
body { margin:0; background:var(--surface); color:var(--ink);
       font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 8px;
     border-bottom:1px solid var(--rule); padding-bottom:4px; }
h3 { font-size:14px; margin:18px 0 6px; color:var(--muted); }
.meta { color:var(--muted); font-size:12px; }
table { border-collapse:collapse; width:100%; margin:6px 0 12px; font-size:13px; }
th, td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--rule);
         vertical-align:top; }
th { color:var(--muted); font-weight:600; }
td.num { font-variant-numeric:tabular-nums; }
.badge { display:inline-block; min-width:64px; padding:1px 8px; border-radius:10px;
         color:#fff; font-weight:600; font-size:12px; }
.badge.warn { color:#0b0b0b; }
.summary span { margin-right:14px; }
figure { margin:12px 0 20px; } figure img { max-width:100%; height:auto; }
figcaption { color:var(--muted); font-size:12px; }
details { margin:8px 0; } pre { white-space:pre-wrap; font-size:12px; color:var(--muted); }
"""


def _badge(status: str) -> str:
    return (
        f'<span class="badge {status}" style="background:{STATUS.get(status, "#52514e")}">'
        f"{STATUS_ICON.get(status, '?')} {status.upper()}</span>"
    )


def _cell(v: Any) -> str:
    if isinstance(v, float):
        return f'<td class="num">{v:.4g}</td>'
    if isinstance(v, dict):
        return "<td>" + html.escape(", ".join(f"{k} {w}" for k, w in v.items())) + "</td>"
    return f"<td>{html.escape('' if v is None else str(v))}</td>"


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(_cell(r.get(c)) for c in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _checks_table(checks: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{_badge(c['status'])}</td><td>{html.escape(c['name'])}</td>"
        f"{_cell(c['value'])}<td>{html.escape(str(c['expected']))}</td>"
        f"<td>{html.escape(c.get('message', ''))}</td></tr>"
        for c in checks
    )
    return (
        "<table><thead><tr><th>status</th><th>check</th><th>value</th><th>expected</th>"
        f"<th>note</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _figure(path: Path) -> str:
    if not path.is_file():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<figure><img alt="{html.escape(path.stem)}" src="data:image/png;base64,{data}">'
        f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
    )


def _extra_tables(stage: str, data: dict[str, Any]) -> str:
    parts = []
    if stage == "sensitivity" and "median_uncertainty" in data:
        rows = [
            {"measure": k, **{q: round(v, 1) for q, v in vals.items()}}
            for k, vals in data["median_uncertainty"].items()
        ]
        parts.append("<h3>Median predicted uncertainty, all candidates</h3>" + _table(rows))
        parts.append("<h3>Station contribution</h3>" + _table(data["station_contribution"]))
    if stage == "ensemble":
        if data.get("heterogeneity_sd"):
            parts.append("<h3>Heterogeneity amplitude</h3>" + _table(data["heterogeneity_sd"]))
        if data.get("correlation_lengths"):
            parts.append("<h3>Correlation lengths</h3>" + _table(data["correlation_lengths"]))
    if stage == "design" and "networks" in data:
        rec = data.get("recommended", {})
        parts.append(
            f"<h3>Recommended network: {rec.get('n')} stations</h3><p>"
            + html.escape(", ".join(rec.get("stations", [])))
            + " (most important first)</p>"
        )
        parts.append("<h3>Best network per size</h3>" + _table(data["networks"]))
        if data.get("margin_sensitivity"):
            parts.append(
                "<h3>Sensitivity to the detection margin</h3>" + _table(data["margin_sensitivity"])
            )
    if stage == "validation" and "summary" in data:
        s = data["summary"]
        rows = [
            {
                "axis": ax,
                **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in s[ax].items()},
            }
            for ax in ("x", "y", "z")
            if ax in s
        ]
        parts.append("<h3>Mislocation statistics</h3>" + _table(rows))
    return "".join(parts)


def build_report(out: Path) -> Path:
    """Write ``out/qc/report.html`` from whatever stage files exist."""
    qdir = out / "qc"
    sections, counts = [], {"pass": 0, "warn": 0, "fail": 0, "info": 0}
    runs = sorted(qdir.glob("run_*.json"))
    run_html = []
    for path in runs:
        data = json.loads(path.read_text())
        for c in data["checks"]:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        v = data["versions"]
        timing = ", ".join(f"{k} {t:.0f} s" for k, t in data["timings_s"].items())
        disk = data.get("peak_disk_mb")
        run_html.append(
            f"<h3>{html.escape(data['stage'])} — finished {html.escape(data['finished'])}</h3>"
            f'<p class="meta">netdesign {v["netdesign"]} · NonLinLoc {v["nonlinloc"]} · '
            f"Python {v['python']} · NumPy {v['numpy']} · SciPy {v['scipy']}<br>"
            f"seeds {html.escape(json.dumps(data['seeds']))}<br>time: {html.escape(timing)}"
            + (f" · peak disk {disk:.0f} MB" if disk is not None else "")
            + "</p>"
            + _checks_table(data["checks"])
            + f"<details><summary>resolved config</summary><pre>"
            f"{html.escape(json.dumps(data['config'], indent=1))}</pre></details>"
        )
    if run_html:
        sections.append("<h2>Run</h2>" + "".join(run_html))
    for stage, title in STAGES:
        path = qdir / f"{stage}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for c in data["checks"]:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        figs = "".join(_figure(qdir / f) for f in data.get("figures", []))
        sections.append(
            f"<h2>{html.escape(title)}</h2>"
            + _checks_table(data["checks"])
            + _extra_tables(stage, data)
            + figs
        )
    summary = "".join(
        f"<span>{_badge(s)} {counts.get(s, 0)}</span>" for s in ("pass", "warn", "fail")
    )
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>netdesign QC report</title><style>{CSS}</style></head><body><main>"
        f'<h1>netdesign QC report</h1><p class="meta">{html.escape(str(out))}</p>'
        f'<p class="summary">{summary}</p>' + "".join(sections) + "</main></body></html>"
    )
    target = qdir / "report.html"
    qdir.mkdir(parents=True, exist_ok=True)
    target.write_text(page)
    return target
