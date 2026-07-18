"""Static leaderboard reports — Markdown + self-contained HTML (stdlib only, no server).

CLI (also exposed as ``connectome-bench-report``)::

    connectome-bench-report ingest --pytest .benchmarks/**/NNNN.json --gbench run.json
    connectome-bench-report report --out results/report.html --target spmm
"""
from __future__ import annotations

import argparse
import glob
import html
from pathlib import Path
from typing import Any

from connectome_dataset.benchmarks.history import ingest, leaderboard, provenance, store


def _fmt(v: Any, spec: str = ".3g") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def to_markdown(lb: dict[str, Any]) -> str:
    lines = ["# Benchmark Leaderboard", "", f"_Baseline provider: `{lb['baseline_provider']}`_", ""]
    lines += ["## Aggregate (geomean across cells)", "",
              "| Rank | Implementation | ×best | ×baseline | Cells |", "|---|---|---|---|---|"]
    for i, a in enumerate(lb["aggregate"], 1):
        base = f"{_fmt(a['geomean_vs_baseline'])}×" if a["geomean_vs_baseline"] is not None else "-"
        lines.append(f"| {i} | `{a['name']}` | {_fmt(a['geomean_vs_best'])}× | {base} | {a['n_cells']} |")
    lines.append("")
    for cell in lb["cells"]:
        lines += [f"### {cell['target']} · {cell['case_id']} · {cell['problem']} · hw={cell['hardware_id']}",
                  "", f"Primary: `{cell['primary_metric']}`", "",
                  "| Impl | Primary | Median ms | ×best | ×baseline |", "|---|---|---|---|---|"]
        for r in cell["ranking"]:
            lines.append(
                f"| `{r['name']}` | {_fmt(r['primary'])} | {_fmt(r['median_ms'])} | "
                f"{_fmt(r['speedup_vs_best'])}× | {_fmt(r['speedup_vs_baseline'])} |"
            )
        lines.append("")
    return "\n".join(lines)


_CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:70rem}
h1{margin-bottom:.2rem}table{border-collapse:collapse;margin:.5rem 0 1.5rem;width:100%}
th,td{border:1px solid #ccc;padding:.3rem .6rem;text-align:right}th:first-child,td:first-child{text-align:left}
th{background:#f4f4f4}code{background:#f0f0f0;padding:0 .2rem;border-radius:3px}
.prov{color:#555;font-size:.85em}tr:nth-child(even){background:#fafafa}
@media(prefers-color-scheme:dark){body{background:#111;color:#ddd}th{background:#222}
tr:nth-child(even){background:#181818}code{background:#222}th,td{border-color:#333}.prov{color:#999}}
"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def to_html(lb: dict[str, Any], prov: dict[str, Any] | None = None) -> str:
    prov = prov or provenance.collect()
    hw, sw, git = prov["hardware"], prov["software"], prov["git"]
    parts = [f"<style>{_CSS}</style>", "<h1>Benchmark Leaderboard</h1>",
             f"<p class='prov'>GPU: {html.escape(str(hw.get('gpu')))} "
             f"(arch {html.escape(str(hw.get('arch')))}, sm {hw.get('sm')}, hw_id {prov['hardware_id']}) · "
             f"CUDA {html.escape(str(sw.get('cuda')))} · driver {html.escape(str(sw.get('driver')))} · "
             f"git {html.escape(str((git.get('commit') or '')[:8]))}"
             f"{'+dirty' if git.get('dirty') else ''} · baseline <code>{html.escape(lb['baseline_provider'])}</code></p>",
             "<h2>Aggregate — geomean across cells</h2>"]
    parts.append(_table(
        ["Rank", "Implementation", "×best", "×baseline", "Cells"],
        [[str(i), f"<code>{html.escape(a['name'])}</code>", f"{_fmt(a['geomean_vs_best'])}×",
          (f"{_fmt(a['geomean_vs_baseline'])}×" if a["geomean_vs_baseline"] is not None else "-"),
          str(a["n_cells"])]
         for i, a in enumerate(lb["aggregate"], 1)],
    ))
    for cell in lb["cells"]:
        parts.append(f"<h3>{html.escape(cell['target'])} · {html.escape(str(cell['case_id']))} · "
                     f"{html.escape(cell['problem'])} · hw={cell['hardware_id']}</h3>")
        parts.append(f"<p class='prov'>primary: <code>{html.escape(cell['primary_metric'])}</code></p>")
        parts.append(_table(
            ["Impl", "Primary", "Median ms", "×best", "×baseline"],
            [[f"<code>{html.escape(r['name'])}</code>", _fmt(r["primary"]), _fmt(r["median_ms"]),
              f"{_fmt(r['speedup_vs_best'])}×",
              (f"{_fmt(r['speedup_vs_baseline'])}×" if r["speedup_vs_baseline"] is not None else "-")]
             for r in cell["ranking"]],
        ))
    return "<!doctype html><meta charset='utf-8'><title>Benchmark Leaderboard</title>" + "".join(parts)


def _cmd_ingest(args: argparse.Namespace) -> None:
    prov = provenance.collect()
    records: list[dict[str, Any]] = []
    for p in args.pytest or []:
        for f in glob.glob(p, recursive=True):
            records += ingest.ingest_pytest_benchmark(f, prov)
    for p in args.gbench or []:
        for f in glob.glob(p, recursive=True):
            records += ingest.ingest_google_benchmark(f, prov)
    for p in args.cpp or []:
        for f in glob.glob(p, recursive=True):
            records += ingest.ingest_connectome_jsonl(f, prov)
    n = store.append(records)
    print(f"Ingested {n} records into {store.HISTORY_PATH}")


def _cmd_report(args: argparse.Namespace) -> None:
    lb = leaderboard.build(store.load(), target=args.target, baseline_provider=args.baseline)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_html(lb))
    print(f"Wrote {out}")
    if args.markdown:
        md = Path(args.markdown)
        md.write_text(to_markdown(lb))
        print(f"Wrote {md}")


def main() -> None:
    p = argparse.ArgumentParser(prog="connectome-bench-report", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="normalize native JSON into results/history.jsonl")
    pi.add_argument("--pytest", nargs="*", help="pytest-benchmark JSON path(s)/glob(s)")
    pi.add_argument("--gbench", nargs="*", help="Google Benchmark JSON path(s)/glob(s)")
    pi.add_argument("--cpp", nargs="*", help="connectome_bench BenchmarkRecord JSONL path(s)/glob(s)")
    pi.set_defaults(func=_cmd_ingest)

    pr = sub.add_parser("report", help="build leaderboard report from history")
    pr.add_argument("--out", default="results/report.html")
    pr.add_argument("--markdown", default="results/leaderboard.md")
    pr.add_argument("--target", default=None, choices=["spmm", "spgemm", "rsnn"])
    pr.add_argument("--baseline", default="cusparse")
    pr.set_defaults(func=_cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
