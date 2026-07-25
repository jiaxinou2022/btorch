"""Out-of-tree provider registry — how third-party sparse kernels plug in.

The custom SOTA kernels (Sputnik, FlashSparse, DTC-SpMM, MH-SpGEMM, VDHA, …) are
developed *out of tree*; this module is the boundary they cross to be benchmarked
against the in-tree baselines through the shared cases, measurement, and record schema.

One registry per **target**, because their operands and validation differ:

- :class:`SpmmProvider`  — SpMM ``C = A·B`` (N=1 is SpMV); group ``connectome_bench.spmm``.
- :class:`SpgemmProvider` — SpGEMM ``C = A·Aᵀ`` (sparse×sparse); group ``connectome_bench.spgemm``.
- :class:`SpMSpVProvider` — SpMSpV ``y = A·x`` (sparse matrix × sparse vector); group
  ``connectome_bench.spmspv``.

Each is discovered two ways, both merged by the matching ``discover_*``:

1. **Entry points** (the out-of-tree path): an installed package declares the target's
   entry-point group pointing at a zero-arg callable returning one or more providers.
   Nothing in this repo needs editing.
2. **In-process registration** (``register_*``): for notebooks / tests / an in-repo
   kernel that hasn't been packaged yet.

A broken entry point warns and is skipped rather than breaking discovery for the
others — one bad third-party package must not take down the whole leaderboard.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable

SPMM_GROUP = "connectome_bench.spmm"
SPGEMM_GROUP = "connectome_bench.spgemm"
SPMSPV_GROUP = "connectome_bench.spmspv"
ENTRY_POINT_GROUP = SPMM_GROUP  # backwards-compatible alias

# make_fn(matrix, *, variant, device, ...) -> (() -> result). The per-target keyword
# operands differ (see each provider's docstring) but the returned callable shape is
# uniform: it runs the op and returns the (device-resident) result.
MakeFn = Callable[..., Callable[[], Any]]


@dataclass(frozen=True)
class _Provider:
    """Shared fields of every target provider (see subclasses for ``make_fn`` operands)."""

    framework: str
    provider: str
    make_fn: MakeFn
    variants: list[str] = field(default_factory=lambda: ["csr"])
    sync: Callable[[], None] | None = None
    supports: Callable[[str, str], bool] | None = None  # (variant, precision) -> bool
    # Some kernels (e.g. FlashSparse, VDHA) manage their own device transfers and return
    # an authoritative kernel-only time from internal CUDA events. When True, the callable
    # returns ``(result, kernel_ms)`` and the harness records that time instead of
    # wall-clock (which would be polluted by the kernel's own H2D/D2H).
    self_timed: bool = False


@dataclass(frozen=True)
class SpmmProvider(_Provider):
    """A benchmarkable SpMM (C = A·B, N=1 is SpMV) implementation.

    ``make_fn(matrix, *, variant, device, batch_size, precision, x_np)`` returns a
    zero-arg callable performing the multiply and returning the dense result; ``x_np``
    is the dense operand the harness also feeds its fp32 oracle.
    """


@dataclass(frozen=True)
class SpgemmProvider(_Provider):
    """A benchmarkable SpGEMM (C = A·Aᵀ, sparse×sparse) implementation.

    ``make_fn(matrix, *, variant, device, precision)`` returns a zero-arg callable that
    computes C and returns a result convertible to a SciPy CSR (via ``providers.to_csr``:
    a ``scipy`` matrix, an object with ``.tocsr()``/``.get_csr()``, or an
    ``(indptr, indices, data, shape)`` tuple) so the oracle can check nnz + values.
    """


@dataclass(frozen=True)
class SpMSpVProvider(_Provider):
    """A benchmarkable SpMSpV (y = A·x, sparse matrix × sparse vector) implementation.

    ``make_fn(matrix, *, variant, device, precision, x_indices, x_values)`` receives the
    sparse operand as parallel index/value arrays and returns a zero-arg callable that
    computes the dense result vector ``y`` (device-resident); the harness validates it
    against a SciPy oracle over the same sparse ``x``.
    """


_SPMM_BUILTINS: list[SpmmProvider] = []
_SPGEMM_BUILTINS: list[SpgemmProvider] = []
_SPMSPV_BUILTINS: list[SpMSpVProvider] = []


def _register(builtins: list, provider: _Provider) -> None:
    key = (provider.framework, provider.provider)
    builtins[:] = [p for p in builtins if (p.framework, p.provider) != key]
    builtins.append(provider)


def _discover(builtins: list, group: str, kind: type) -> list:
    found = list(builtins)
    for ep in metadata.entry_points(group=group):
        try:
            obj = ep.load()
            result = obj() if callable(obj) else obj
            items = result if isinstance(result, (list, tuple)) else [result]
            found.extend(p for p in items if isinstance(p, kind))
        except Exception as e:  # noqa: BLE001 — isolate third-party import/registration failures
            warnings.warn(f"skipping {group} provider entry point {ep.name!r}: {e}", stacklevel=2)
    return found


def register_spmm(provider: SpmmProvider) -> None:
    """Register an SpMM provider in-process (idempotent by (framework, provider))."""
    _register(_SPMM_BUILTINS, provider)


def register_spgemm(provider: SpgemmProvider) -> None:
    """Register an SpGEMM provider in-process (idempotent by (framework, provider))."""
    _register(_SPGEMM_BUILTINS, provider)


def register_spmspv(provider: SpMSpVProvider) -> None:
    """Register an SpMSpV provider in-process (idempotent by (framework, provider))."""
    _register(_SPMSPV_BUILTINS, provider)


def discover_spmm() -> list[SpmmProvider]:
    """All SpMM providers: in-process builtins + installed entry points."""
    return _discover(_SPMM_BUILTINS, SPMM_GROUP, SpmmProvider)


def discover_spgemm() -> list[SpgemmProvider]:
    """All SpGEMM providers: in-process builtins + installed entry points."""
    return _discover(_SPGEMM_BUILTINS, SPGEMM_GROUP, SpgemmProvider)


def discover_spmspv() -> list[SpMSpVProvider]:
    """All SpMSpV providers: in-process builtins + installed entry points."""
    return _discover(_SPMSPV_BUILTINS, SPMSPV_GROUP, SpMSpVProvider)


def to_numpy(result: Any):
    """Best-effort host copy of a provider result (numpy / torch / cupy / array-like)."""
    import numpy as np

    if hasattr(result, "detach"):  # torch
        return result.detach().to("cpu").numpy()
    if hasattr(result, "get") and not hasattr(result, "tocsr"):  # cupy dense (not a sparse mat)
        return result.get()
    return np.asarray(result)


def to_csr(result: Any):
    """Best-effort host copy of an SpGEMM result as a SciPy CSR matrix.

    Accepts a SciPy/cupy sparse matrix, any object exposing ``tocsr()`` or ``get_csr()``,
    or an ``(indptr, indices, data, shape)`` tuple (the ctypes providers' shape).
    """
    import numpy as np
    import scipy.sparse as sp

    if hasattr(result, "get_csr"):
        result = result.get_csr()
    if sp.issparse(result):
        return result.tocsr()
    if hasattr(result, "tocsr"):  # cupy sparse -> host
        host = result.get() if hasattr(result, "get") else result
        return host.tocsr()
    indptr, indices, data, shape = result
    return sp.csr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=shape)
