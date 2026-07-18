"""Unit tests for the out-of-tree provider registry."""
from __future__ import annotations

import numpy as np

from connectome_dataset.benchmarks import providers


def _dummy(**_kwargs):
    return lambda: np.zeros((2, 2), dtype=np.float32)


def test_register_and_discover(monkeypatch) -> None:
    monkeypatch.setattr(providers, "_BUILTINS", [])
    p = providers.SpmmProvider(framework="test", provider="dummy", make_fn=_dummy)
    providers.register_spmm(p)
    found = providers.discover_spmm()
    assert p in found


def test_register_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(providers, "_BUILTINS", [])
    providers.register_spmm(providers.SpmmProvider(framework="t", provider="d", make_fn=_dummy, variants=["csr"]))
    providers.register_spmm(providers.SpmmProvider(framework="t", provider="d", make_fn=_dummy, variants=["coo"]))
    matches = [p for p in providers.discover_spmm() if (p.framework, p.provider) == ("t", "d")]
    assert len(matches) == 1
    assert matches[0].variants == ["coo"]  # the later registration wins


def test_to_numpy_passthrough() -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(providers.to_numpy(arr), arr)


def test_broken_entry_point_is_skipped(monkeypatch) -> None:
    class _BadEP:
        name = "broken"

        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(providers, "_BUILTINS", [])
    monkeypatch.setattr(providers.metadata, "entry_points", lambda group=None: [_BadEP()])
    # discovery must not raise; the broken provider is warned-and-skipped
    import pytest

    with pytest.warns(UserWarning, match="skipping SpMM provider"):
        assert providers.discover_spmm() == []
