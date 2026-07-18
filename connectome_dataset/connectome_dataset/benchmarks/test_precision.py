"""Unit tests for the precision knob's reference-oracle validation."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from connectome_dataset.benchmarks import precision as prec


def _case(seed: int = 0):
    rng = np.random.default_rng(seed)
    a = sp.random(200, 200, density=0.05, format="csr", dtype=np.float32, random_state=seed)
    x = rng.standard_normal((200, 8), dtype=np.float32)
    return a, x


def test_exact_fp32_validates() -> None:
    a, x = _case()
    y = prec.oracle_spmm(a, x)
    assert prec.validate_spmm(y, a, x, "fp32")


def test_wrong_result_rejected() -> None:
    a, x = _case()
    y = prec.oracle_spmm(a, x) + 10.0  # grossly wrong
    assert not prec.validate_spmm(y, a, x, "fp32")


def test_half_precision_result_within_tolerance() -> None:
    a, x = _case()
    y16 = (a.astype(np.float32) @ x).astype(np.float16).astype(np.float32)  # simulate fp16 round-off
    assert prec.validate_spmm(y16, a, x, "fp16")
    # the same round-off must NOT pass under the strict fp32 tolerance
    assert not prec.validate_spmm(y16, a, x, "fp32")


def test_tolerances_ordered() -> None:
    assert prec.rtol("fp32") < prec.rtol("fp16") <= prec.rtol("bf16")
