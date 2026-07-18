"""Unit tests for named matrix sets."""
from __future__ import annotations

import pytest

from connectome_dataset.benchmarks import corpora
from connectome_dataset.catalog import find_graph


def test_named_set_resolves() -> None:
    graphs = corpora.resolve_matrix_set("connectome")
    assert graphs == corpora.CONNECTOME_REPRESENTATIVE
    assert len(graphs) == len(set(graphs))  # no duplicates


def test_single_graph_passthrough() -> None:
    assert corpora.resolve_matrix_set("mice_column_v1") == ["mice_column_v1"]


def test_unknown_rejected() -> None:
    with pytest.raises(ValueError, match="unknown matrix set"):
        corpora.resolve_matrix_set("does_not_exist_xyz")


def test_curated_graphs_exist_in_catalog() -> None:
    """Every graph in every curated set must be a real, loadable catalog entry."""
    for name, graphs in corpora.MATRIX_SETS.items():
        missing = [g for g in graphs if find_graph(g) is None]
        assert not missing, f"{name}: curated graphs missing from catalog: {missing}"


def test_prefix_glob_expands() -> None:
    newman = corpora.resolve_matrix_set("suitesparse_newman_*")
    assert newman and all(g.startswith("suitesparse_newman_") for g in newman)
    assert newman == sorted(newman)  # deterministic order
    # a subset of the full suitesparse prefix
    assert set(newman).issubset(set(corpora.resolve_matrix_set("suitesparse_*")))


def test_prefix_glob_no_match_rejected() -> None:
    with pytest.raises(ValueError, match="no catalog graphs match prefix"):
        corpora.resolve_matrix_set("nonexistent_prefix_*")
