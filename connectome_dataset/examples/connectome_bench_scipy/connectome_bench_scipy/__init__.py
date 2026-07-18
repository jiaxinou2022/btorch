"""Example out-of-tree SpMM provider: a plain SciPy CSR multiply.

This is a template for wrapping a real custom kernel (Sputnik, FlashSparse, …): it
implements the ``make_fn`` contract and advertises itself through the
``connectome_bench.spmm`` entry point declared in ``pyproject.toml``. Once
``pip install``-ed, ``connectome_dataset.benchmarks.providers.discover_spmm`` finds
it and it runs on the shared cases / leaderboard.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from connectome_dataset.benchmarks.providers import SpmmProvider


def _make_fn(
    matrix: sp.csr_matrix, *, variant: str, device: str, batch_size: int,
    precision: str = "fp32", x_np: np.ndarray | None = None,
) -> Callable[[], Any]:
    a = matrix.tocsr().astype(np.float32)
    if x_np is None:
        x_np = np.random.default_rng(0).random((a.shape[0], batch_size), dtype=np.float32)
    x = x_np.astype(np.float32)
    return lambda: a @ x


def get_providers() -> list[SpmmProvider]:
    return [
        SpmmProvider(
            framework="scipy",
            provider="scipy_ref",
            make_fn=_make_fn,
            variants=["csr"],
            supports=lambda variant, precision: precision == "fp32",  # CPU reference, fp32 only
        )
    ]
