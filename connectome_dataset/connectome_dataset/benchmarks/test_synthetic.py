"""Unit tests for the synthetic controlled-sweep generators."""
from __future__ import annotations

import numpy as np
import pytest

from connectome_dataset.benchmarks import synthetic
from connectome_dataset.benchmarks.cases import stratification

N = 2048
DEG = 16.0


def _degree_cov(matrix) -> float:
    deg = np.diff(matrix.indptr).astype(float)
    return float(deg.std() / deg.mean()) if deg.mean() > 0 else 0.0


@pytest.mark.parametrize("family", list(synthetic._GENERATORS))
def test_shape_and_determinism(family: str) -> None:
    gen = synthetic._GENERATORS[family]
    a = gen(N, DEG, seed=0)
    b = gen(N, DEG, seed=0)
    assert a.shape == (N, N)
    # same seed -> byte-identical structure and values
    assert (a != b).nnz == 0
    # a different seed changes the draw (banded structure is fixed, so skip it there)
    if family != "banded":
        c = gen(N, DEG, seed=1)
        assert (a != c).nnz > 0


@pytest.mark.parametrize("family", list(synthetic._GENERATORS))
def test_no_self_loops_and_degree_budget(family: str) -> None:
    m = synthetic._GENERATORS[family](N, DEG, seed=0)
    assert m.diagonal().sum() == 0.0  # self-loops dropped
    # dedup can only remove edges, so realized avg degree never exceeds the target
    assert m.nnz <= N * DEG
    assert m.nnz > 0


def test_cov_ordering() -> None:
    """banded (structured) < erdos_renyi (uniform) < power_law (heavy-tailed)."""
    banded = _degree_cov(synthetic.banded(N, DEG, seed=0))
    er = _degree_cov(synthetic.erdos_renyi(N, DEG, seed=0))
    pl = _degree_cov(synthetic.power_law(N, DEG, gamma=2.1, seed=0))
    assert banded < er < pl


def test_heavier_tail_raises_cov() -> None:
    lighter = _degree_cov(synthetic.power_law(N, DEG, gamma=3.0, seed=0))
    heavier = _degree_cov(synthetic.power_law(N, DEG, gamma=2.1, seed=0))
    assert heavier > lighter


def test_banded_bandwidth() -> None:
    half = max(1, int(round(DEG / 2.0)))
    m = synthetic.banded(N, DEG, seed=0).tocoo()
    assert np.all(np.abs(m.row - m.col) <= half)


def test_spec_name_and_generate() -> None:
    spec = synthetic.SyntheticSpec("power_law", 512, 8.0, params={"gamma": 2.5})
    assert "power_law" in spec.name and "gamma2.5" in spec.name
    assert spec.generate().shape == (512, 512)


def test_unknown_family_rejected() -> None:
    with pytest.raises(ValueError, match="unknown synthetic family"):
        synthetic.SyntheticSpec("nope", 16).generate()


def test_cases_flow_through_stratification() -> None:
    spec = synthetic.SyntheticSpec("power_law", N, DEG, params={"gamma": 2.1})
    case = synthetic.spmv_case(spec)
    assert case.n == N
    assert case.case_id.endswith("_spmv")
    assert case.metadata["synthetic"] is True
    strat = stratification(case)
    # avg_degree ~ DEG places it in Voltrix Type I (<20); heavy tail -> nonzero Gini
    assert strat["degree_type"] == "I"
    assert strat["degree_gini"] > 0.0
    assert strat["degree_cov"] > 0.0


def test_spgemm_case_has_intermediate_products() -> None:
    spec = synthetic.SyntheticSpec("erdos_renyi", 512, 8.0)
    case = synthetic.spgemm_case(spec)
    assert case.intermediate_products > 0
    assert case.case_id.endswith("_spgemm")


def test_default_sweep_shares_size() -> None:
    specs = synthetic.default_sweep(n=1024, avg_degree=12.0)
    assert len(specs) == 5
    assert {s.family for s in specs} == {"banded", "erdos_renyi", "power_law", "rmat"}
    for s in specs:
        assert s.n == 1024 and s.avg_degree == 12.0
