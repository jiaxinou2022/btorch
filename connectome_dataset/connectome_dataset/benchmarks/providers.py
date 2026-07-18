"""Out-of-tree provider registry — how third-party SpMM kernels plug in.

The custom SOTA kernels (Sputnik, FlashSparse, DTC-SpMM, …) are developed *out of
tree*; this module is the boundary they cross to be benchmarked against the in-tree
baselines through the shared cases, measurement, and record schema.

A provider is discovered two ways, both merged by :func:`discover_spmm`:

1. **Entry points** (the out-of-tree path): an installed package declares
   ``[project.entry-points."connectome_bench.spmm"]`` pointing at a zero-arg callable
   that returns one or more :class:`SpmmProvider`. Nothing in this repo needs editing.
2. **In-process registration** (:func:`register_spmm`): for notebooks / tests / an
   in-repo kernel that hasn't been packaged yet.

This replaces the old in-tree ``registry.py`` (a global-mutable dead stub): the
registry *is* the discovery mechanism, and its reason to exist is the out-of-tree
boundary, not a second way to wire in-tree providers.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable

ENTRY_POINT_GROUP = "connectome_bench.spmm"

# make_fn(matrix, *, variant, device, batch_size, precision, x_np) -> (() -> result)
MakeFn = Callable[..., Callable[[], Any]]


@dataclass(frozen=True)
class SpmmProvider:
    """A benchmarkable SpMM (C = A·B, N=1 is SpMV) implementation.

    ``make_fn`` receives the SciPy ``matrix`` plus ``variant``, ``device``,
    ``batch_size``, ``precision`` and the dense operand ``x_np`` (so the harness can
    validate the result against its fp32 oracle) and returns a zero-arg callable that
    performs the multiply and returns the (device-resident) result.
    """

    framework: str
    provider: str
    make_fn: MakeFn
    variants: list[str] = field(default_factory=lambda: ["csr"])
    sync: Callable[[], None] | None = None
    supports: Callable[[str, str], bool] | None = None  # (variant, precision) -> bool
    # Some kernels (e.g. FlashSparse) manage their own device transfers and return an
    # authoritative kernel-only time from internal CUDA events. When True, the callable
    # returns ``(result, kernel_ms)`` and the harness records that time instead of
    # wall-clock (which would be polluted by the kernel's own H2D/D2H).
    self_timed: bool = False


_BUILTINS: list[SpmmProvider] = []


def register_spmm(provider: SpmmProvider) -> None:
    """Register a provider in-process (idempotent by (framework, provider))."""
    _BUILTINS[:] = [p for p in _BUILTINS if (p.framework, p.provider) != (provider.framework, provider.provider)]
    _BUILTINS.append(provider)


def discover_spmm() -> list[SpmmProvider]:
    """All registered providers: in-process builtins + installed entry points.

    A broken entry point warns and is skipped rather than breaking discovery for the
    others — one bad third-party package must not take down the whole leaderboard.
    """
    found = list(_BUILTINS)
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
            result = obj() if callable(obj) else obj
            items = result if isinstance(result, (list, tuple)) else [result]
            found.extend(p for p in items if isinstance(p, SpmmProvider))
        except Exception as e:  # noqa: BLE001 — isolate third-party import/registration failures
            warnings.warn(f"skipping SpMM provider entry point {ep.name!r}: {e}", stacklevel=2)
    return found


def to_numpy(result: Any):
    """Best-effort host copy of a provider result (numpy / torch / cupy / array-like)."""
    import numpy as np

    if hasattr(result, "detach"):  # torch
        return result.detach().to("cpu").numpy()
    if hasattr(result, "get"):  # cupy
        return result.get()
    return np.asarray(result)
