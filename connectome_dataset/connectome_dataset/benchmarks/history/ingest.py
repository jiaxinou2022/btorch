"""Adapters mapping native benchmark JSON into the unified record schema.

Three producers, one schema (docs/benchmark_design/history_and_reporting.md):
- pytest-benchmark: ``benchmarks[].{stats, extra_info, params}`` — identity comes
  from our ``extra_info`` (populated by templates.base_extra_info).
- Google Benchmark: ``benchmarks[].{name, real_time, counters}`` — identity is
  parsed from the ``<provider>/<variant>/<target>`` name hierarchy.
- connectome_bench (C++ runner): one BenchmarkRecord per JSONL line.

All three are normalized through :func:`_canonical_target` and :func:`_canonical_problem`
so their records land in the same leaderboard cell (the whole point of unification).

Provenance is currently collected from the *ingesting* host, which is correct only when
you report on the machine that produced the runs (the common case). The C++ manifest and
gbench context each record their producing host; wiring that through so cross-machine
aggregation stamps produce-time provenance is a follow-up.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from connectome_dataset.benchmarks.history import provenance
from connectome_dataset.metrics import real_time_factor, spikes_per_s, synaptic_events_per_s, tflops

# Per-phase wall-time breakdown carried through from the beNNch RSNN leaf into metrics.
_RSNN_PHASE_KEYS = (
    "phase_full_ms", "phase_deliver_ms", "phase_update_ms",
    "phase_gather_ms", "phase_communicate_ms", "build_ms",
)

SCHEMA_VERSION = 1


def _canonical_target(target: str | None) -> str | None:
    # N=1 SpMV is the batch=1 case of SpMM; one canonical target name everywhere.
    return "spmm" if target == "spmv" else target


_PRECISION_FROM_DTYPE = {
    "float32": "fp32", "fp32": "fp32",
    "float16": "fp16", "fp16": "fp16",
    "bfloat16": "bf16", "bf16": "bf16",
}


def _precision_knob(dtype: str | None) -> dict[str, Any]:
    # C++ records carry precision as the operand dtype; the leaderboard keys cells on
    # knobs.precision, so fold it in to align with the Python providers' precision knob.
    p = _PRECISION_FROM_DTYPE.get((dtype or "").lower())
    return {"precision": p} if p and p != "fp32" else {}


def _canonical_problem(problem: dict[str, Any] | None, target: str | None) -> dict[str, Any]:
    """Reduce a producer's problem dict to the minimal cell-defining axes.

    spmm/spgemm are keyed on dense width ``N`` (``batch_size``/``bs`` are synonyms);
    rsnn is keyed on ``timesteps`` and ``batch_size``.
    """
    p = dict(problem or {})
    if target in ("spmm", "spgemm"):
        n = p.get("N", p.get("batch_size", p.get("bs")))
        return {"N": n} if n is not None else {}
    # rsnn: fwd vs fwd+bwd is a distinct workload, so `pass` is part of the cell.
    return {k: p[k] for k in ("timesteps", "batch_size", "pass") if p.get(k) is not None}


def _record(
    *, source: str, run_id: str, timestamp: str, identity: dict[str, Any],
    metrics: dict[str, Any], prov: dict[str, Any], status: str = "ok",
) -> dict[str, Any]:
    identity = dict(identity)
    identity["target"] = _canonical_target(identity.get("target"))
    identity["problem"] = _canonical_problem(identity.get("problem"), identity["target"])
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": timestamp,
        "source": source,
        **identity,
        "metrics": metrics,
        "status": status,
        "hardware": prov["hardware"],
        "hardware_id": prov["hardware_id"],
        "software": prov["software"],
        "git": prov["git"],
    }


def _stats_metrics(stats: dict[str, Any], extra: dict[str, Any], target: str | None) -> dict[str, Any]:
    # A self-timed kernel (e.g. FlashSparse) reports its own transfer-free time; prefer it
    # over pytest-benchmark's wall-clock, which would include the kernel's internal H2D/D2H.
    median_ms = extra["self_timed_ms"] if extra.get("self_timed_ms") else stats.get("median", 0.0) * 1000.0
    total_flops = extra.get("total_flops")
    m: dict[str, Any] = {
        "compute_ms_median": median_ms,
        "compute_ms_min": stats.get("min", 0.0) * 1000.0,
        "compute_ms_max": stats.get("max", 0.0) * 1000.0,
        "compute_ms_p25": stats.get("q1", 0.0) * 1000.0 if "q1" in stats else None,
        "compute_ms_p75": stats.get("q3", 0.0) * 1000.0 if "q3" in stats else None,
        "compile_ms": extra.get("compile_ms"),
        "preprocess_ms": extra.get("preprocess_ms"),
    }
    if total_flops and median_ms > 0:
        m["tflops"] = tflops(total_flops, median_ms)
    if _canonical_target(target) == "rsnn":
        timesteps = (extra.get("problem") or {}).get("timesteps") or extra.get("timesteps")
        dt_ms = extra.get("dt_ms", 1.0)
        if timesteps and median_ms > 0:
            m["real_time_factor"] = real_time_factor(median_ms, timesteps * dt_ms)
        # beNNch event-rate metrics: counts (over the whole run) / measured hot-loop time.
        wall_s = median_ms * 1e-3
        if extra.get("n_spikes") is not None and median_ms > 0:
            m["spikes_per_s"] = spikes_per_s(extra["n_spikes"], wall_s)
        if extra.get("n_syn_events") is not None and median_ms > 0:
            m["synaptic_events_per_s"] = synaptic_events_per_s(extra["n_syn_events"], wall_s)
        if extra.get("firing_rate_hz") is not None:
            m["firing_rate_hz"] = extra["firing_rate_hz"]
        for k in _RSNN_PHASE_KEYS:
            if extra.get(k) is not None:
                m[k] = extra[k]
    return m


def ingest_pytest_benchmark(path: str | Path, prov: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    prov = prov or provenance.collect()
    run_id = Path(path).stem
    timestamp = data.get("datetime") or data.get("commit_info", {}).get("time", "")
    out = []
    for b in data.get("benchmarks", []):
        extra = b.get("extra_info", {})
        target = extra.get("target")
        identity = {
            "framework": extra.get("framework", "python"),
            "provider": extra.get("provider") or extra.get("alg"),
            "target": target,
            "variant": extra.get("variant"),
            "knobs": extra.get("knobs", {}),
            "case_id": extra.get("case_id"),
            "graph_id": extra.get("graph_id"),
            "n": extra.get("n"),
            "nnz": extra.get("nnz"),
            "density": extra.get("density"),
            "stratification": extra.get("stratification", {}),
            "problem": extra.get("problem", {}),
            "device": extra.get("device"),
        }
        out.append(_record(
            source="pytest-benchmark", run_id=run_id, timestamp=timestamp,
            identity=identity, metrics=_stats_metrics(b.get("stats", {}), extra, target),
            prov=prov, status=extra.get("status", "ok"),
        ))
    return out


def ingest_google_benchmark(path: str | Path, prov: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    prov = prov or provenance.collect()
    run_id = Path(path).stem
    ctx = data.get("context", {})
    timestamp = ctx.get("date", "")
    out = []
    for b in data.get("benchmarks", []):
        if b.get("run_type") == "aggregate":  # keep the raw iterations, skip gbench aggregates
            continue
        counters = {k: v for k, v in b.items() if k not in _GBENCH_RESERVED}
        provider, variant, target = _parse_gbench_name(b.get("name", ""))
        unit_scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}.get(b.get("time_unit", "ns"), 1e-6)
        metrics = {"compute_ms_median": b.get("real_time", 0.0) * unit_scale}
        for gk, mk in (("GFLOP/s", "gflops"), ("GB/s", "gbps")):
            if gk in counters:
                metrics[mk] = counters[gk]
        if "GFLOP/s" in counters:
            metrics["tflops"] = counters["GFLOP/s"] / 1000.0
        identity = {
            "framework": "cpp", "provider": provider, "target": target, "variant": variant,
            "knobs": {}, "case_id": counters.get("case_id"), "graph_id": counters.get("graph_id"),
            "n": counters.get("n"), "nnz": counters.get("nnz"), "density": counters.get("density"),
            "stratification": {}, "problem": {k: counters[k] for k in ("bs", "N") if k in counters},
            "device": ctx.get("device", "cpu"),
        }
        out.append(_record(
            source="google-benchmark", run_id=run_id, timestamp=timestamp,
            identity=identity, metrics=metrics, prov=prov,
        ))
    return out


def ingest_connectome_jsonl(path: str | Path, prov: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Ingest the C++ runner's BenchmarkRecord JSONL (one record per line)."""
    prov = prov or provenance.collect()
    path = Path(path)
    run_id = path.stem
    timestamp = _manifest_created_at(path)
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        provider, _, name_variant = (r.get("alg") or "").partition(".")
        gflops = r.get("gflops_s")
        metrics = {
            "compute_ms_median": r.get("median_ms"),
            "compute_ms_min": r.get("min_ms"),
            "compute_ms_max": r.get("max_ms"),
            "compute_ms_p25": r.get("p25_ms"),
            "compute_ms_p75": r.get("p75_ms"),
            "compile_ms": r.get("compile_ms"),
            "gbps": r.get("gbps"),
            "tflops": gflops / 1000.0 if gflops is not None else None,
        }
        identity = {
            "framework": "cpp", "provider": provider or r.get("alg"),
            "target": r.get("target"), "variant": r.get("variant") or name_variant,
            "knobs": _precision_knob(r.get("dtype")),
            "case_id": r.get("case_id"), "graph_id": r.get("graph_id"),
            "n": r.get("n"), "nnz": r.get("nnz"), "density": r.get("density"),
            "stratification": {},
            "problem": {k: r[k] for k in ("batch_size", "timesteps") if r.get(k) is not None},
            "device": r.get("device"),
        }
        out.append(_record(
            source="connectome-cpp", run_id=run_id, timestamp=timestamp,
            identity=identity, metrics=metrics, prov=prov, status=r.get("status", "ok"),
        ))
    return out


def _manifest_created_at(records_path: Path) -> str:
    manifest = records_path.parent / "manifest.json"
    if manifest.exists():
        try:
            return json.loads(manifest.read_text()).get("created_at", "")
        except (json.JSONDecodeError, OSError):
            return ""
    return ""


_GBENCH_RESERVED = {
    "name", "family_index", "per_family_instance_index", "run_name", "run_type",
    "repetitions", "repetition_index", "threads", "iterations", "real_time", "cpu_time",
    "time_unit", "aggregate_name", "aggregate_unit",
}


def _parse_gbench_name(name: str) -> tuple[str | None, str | None, str | None]:
    # convention: "<provider>/<variant>/<target>" e.g. "eigen/csr/spmv"; drop any "/bs=.." suffix
    base = [p for p in name.split("/", 3) if "=" not in p]
    provider = base[0] if len(base) > 0 else None
    variant = base[1] if len(base) > 1 else None
    target = base[2] if len(base) > 2 else None
    return provider, variant, target
