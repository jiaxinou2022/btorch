"""Leaderboard: rank implementations within a workload cell, aggregate by geomean.

A *workload cell* is the apples-to-apples tuple ``(target, case_id, problem, hardware_id)``
— only implementations in the same cell are comparable. Primary metric by target:
SpMM/SpGEMM → TFLOP/s (desc); RSNN → real-time factor (asc). Rankings show speedup vs
the fastest in the cell and vs a designated baseline provider (default ``cusparse``).
Only correctness-validated (``status == ok``) records are ranked.
"""
from __future__ import annotations

import math
from typing import Any

# (metric key, higher_is_better) per target
_PRIMARY = {
    "spmm": ("tflops", True),
    "spgemm": ("tflops", True),
    "rsnn": ("real_time_factor", False),
}
_FALLBACK = ("compute_ms_median", False)  # lower is better


def _impl_name(r: dict[str, Any]) -> str:
    return f"{r.get('framework')}.{r.get('provider')}.{r.get('target')}.{r.get('variant')}"


def _problem_sig(r: dict[str, Any]) -> str:
    p = r.get("problem") or {}
    knobs = r.get("knobs") or {}
    parts = [f"{k}={p[k]}" for k in sorted(p)]
    # fp32 is the implicit default so records that don't tag precision (legacy runs,
    # C++ fp32) share the fp32 cell; only reduced precision splits off into its own.
    prec = knobs.get("precision")
    if prec and prec != "fp32":
        parts.append(f"precision={prec}")
    return ",".join(parts) or "-"


def _cell_key(r: dict[str, Any]) -> tuple:
    # Fall back to matrix size when there is no case_id (e.g. Google Benchmark
    # counters carry no identifier) so different-sized matrices don't collide.
    case = r.get("case_id") or f"n{r.get('n')}_nnz{r.get('nnz')}"
    return (r.get("target"), case, _problem_sig(r), r.get("hardware_id"))


def _cell_metric(target: str | None, records: list[dict[str, Any]]) -> tuple[str, bool]:
    """The single metric every record in a cell is ranked by.

    Use the target's primary metric only if *every* record has it; otherwise fall
    back to median time for all — never rank one record on tflops and another on
    median (opposite directions, incomparable).
    """
    key, hib = _PRIMARY.get(target, _FALLBACK)
    if all(r.get("metrics", {}).get(key) is not None for r in records):
        return key, hib
    return _FALLBACK


def _latest_per_impl(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple, dict[str, Any]] = {}
    for r in records:
        k = (_impl_name(r), _cell_key(r))
        if k not in best or (r.get("timestamp", "") > best[k].get("timestamp", "")):
            best[k] = r
    return list(best.values())


def build(
    records: list[dict[str, Any]], *, target: str | None = None, baseline_provider: str = "cusparse"
) -> dict[str, Any]:
    records = [r for r in records if r.get("status", "ok") == "ok"]
    if target:
        records = [r for r in records if r.get("target") == target]
    records = _latest_per_impl(records)

    cells: dict[tuple, list[dict[str, Any]]] = {}
    for r in records:
        cells.setdefault(_cell_key(r), []).append(r)

    cell_reports = []
    # Keep the two ratios separate — vs-best (always defined, <=1) and vs-baseline
    # (only for cells containing the baseline) must never be blended into one geomean.
    agg_best: dict[str, list[float]] = {}
    agg_base: dict[str, list[float]] = {}
    for key, rs in sorted(cells.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        target_ = key[0]
        metric_key, hib = _cell_metric(target_, rs)
        scored = [(r, r.get("metrics", {}).get(metric_key)) for r in rs]
        scored = [(r, float(v)) for r, v in scored if v is not None]
        if not scored:
            continue
        scored.sort(key=lambda rv: rv[1], reverse=hib)
        best_val = scored[0][1]
        baseline = next((v for r, v in scored if r.get("provider") == baseline_provider), None)

        ranking = []
        for r, val in scored:
            spd_best = (val / best_val) if hib else (best_val / val)
            spd_base = None
            if baseline:
                spd_base = (val / baseline) if hib else (baseline / val)
            name = _impl_name(r)
            ranking.append({
                "name": name, "provider": r.get("provider"), "variant": r.get("variant"),
                "primary": val, "median_ms": r.get("metrics", {}).get("compute_ms_median"),
                "speedup_vs_best": spd_best, "speedup_vs_baseline": spd_base,
            })
            agg_best.setdefault(name, []).append(spd_best)
            if spd_base is not None:
                agg_base.setdefault(name, []).append(spd_base)

        _, case_id, problem, hw = key
        cell_reports.append({
            "target": target_, "case_id": case_id, "problem": problem, "hardware_id": hw,
            "primary_metric": metric_key, "ranking": ranking,
        })

    aggregate = [
        {
            "name": n,
            "geomean_vs_best": _geomean(bests),
            "geomean_vs_baseline": _geomean(agg_base[n]) if n in agg_base else None,
            "n_cells": len(bests),
        }
        for n, bests in agg_best.items()
    ]
    aggregate.sort(key=lambda a: a["geomean_vs_best"], reverse=True)
    return {"cells": cell_reports, "aggregate": aggregate, "baseline_provider": baseline_provider}


def _geomean(values: list[float]) -> float:
    vals = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
