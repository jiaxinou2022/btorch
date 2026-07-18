"""Reusable pytest-benchmark bodies and the unified ``extra_info`` schema.

Every benchmark leaf populates ``benchmark.extra_info`` through :func:`base_extra_info`
so the identity ``(framework, provider, target, variant)``, the workload
stratification, and the four metric kinds (design.md §1b) are recorded uniformly and
are ingestible into the shared history (history_and_reporting.md). Framework-specific
concerns (JAX ``block_until_ready`` vs ``torch.cuda.synchronize``) stay in the leaf via
the ``sync`` callback — the template does not hide them.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from connectome_dataset.benchmarks.cases import stratification


def measure_compile_ms(fn: Callable[[], Any], sync: Callable[[], None] | None = None) -> float:
    """Time the first (compiling / allocating) call, kept out of the hot loop."""
    t0 = time.perf_counter()
    fn()
    if sync is not None:
        sync()
    return (time.perf_counter() - t0) * 1000.0


def base_extra_info(
    *,
    framework: str,
    provider: str,
    target: str,
    variant: str,
    case: Any,
    device: str,
    problem: dict[str, Any] | None = None,
    knobs: dict[str, Any] | None = None,
    total_flops: int | None = None,
    timings: dict[str, float] | None = None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the unified record fields for ``benchmark.extra_info``.

    ``status`` is ``"ok"`` by default; reference-oracle validation (Phase 2) sets it
    to ``"incorrect"`` so the leaderboard can exclude a kernel that fails correctness.
    """
    info: dict[str, Any] = {
        "framework": framework,
        "provider": provider,
        "target": target,
        "variant": variant,
        "status": status,
        "case_id": getattr(case, "case_id", None),
        "graph_id": getattr(case, "graph_id", None),
        "n": getattr(case, "n", None),
        "nnz": getattr(case, "nnz", None),
        "density": getattr(case, "density", None),
        "device": str(device),
        "stratification": stratification(case),
        # legacy key kept so existing analysis/autosave stays readable:
        "alg": provider,
    }
    if problem:
        info["problem"] = problem
        for k in ("batch_size", "N", "timesteps"):
            if k in problem:
                info[k] = problem[k]
    if knobs:
        info["knobs"] = knobs
    if total_flops is not None:
        info["total_flops"] = total_flops
    if timings:
        info.update(timings)  # e.g. compile_ms, preprocess_ms, load_ms
    if extra:
        info.update(extra)
    return info


def run_pedantic(
    benchmark: Any,
    run: Callable[[], Any],
    *,
    warmup: int,
    rounds: int,
    extra_info: dict[str, Any],
) -> None:
    """Record metadata then run the hot loop under pytest-benchmark."""
    benchmark.extra_info.update(extra_info)
    benchmark.pedantic(run, warmup_rounds=warmup, rounds=rounds)
