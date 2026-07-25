"""Unit tests for the SpGEMM and SpMSpV target plumbing (oracles, cases, ingest cells).

These exercise the CPU-side contract that the MH-SpGEMM and VDHA GPU kernels are held to:
the reference oracles they are validated against, and the leaderboard-cell canonicalization
that decides which providers are ranked head-to-head.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from connectome_dataset.benchmarks import cases, precision as prec
from connectome_dataset.benchmarks.history import ingest
from connectome_dataset.metrics import flops_spmspv


def _rand(n=128, density=0.06, seed=0):
    return sp.random(n, n, density=density, format="csr", random_state=seed).astype(np.float32)


# --- SpGEMM oracle ---------------------------------------------------------------

def test_spgemm_oracle_matches_scipy_and_rejects_perturbation():
    a = _rand()
    c = prec.oracle_spgemm(a)
    assert c.shape == (a.shape[0], a.shape[0])
    assert prec.validate_spgemm(c, a, "fp64")
    bad = c.copy()
    bad.data = bad.data + 1.0
    assert not prec.validate_spgemm(bad, a, "fp64")


def test_spgemm_validate_tolerates_explicit_zeros():
    a = _rand()
    c = prec.oracle_spgemm(a).tocsr()
    # A kernel that emits extra structural zeros (values kept in the pattern but == 0)
    # must still validate — validate_spgemm eliminates zeros before comparing.
    coo = c.tocoo()
    row = np.append(coo.row, 3).astype(np.int64)
    col = np.append(coo.col, 5).astype(np.int64)
    data = np.append(coo.data, 0.0)
    c2 = sp.coo_matrix((data, (row, col)), shape=c.shape).tocsr()
    assert c2.nnz >= c.nnz  # carries the explicit zero before elimination
    assert prec.validate_spgemm(c2, a, "fp64")


# --- SpMSpV oracle ---------------------------------------------------------------

def test_make_sparse_vector_shape_and_determinism():
    idx1, val1 = prec.make_sparse_vector(1000, 0.05, seed=3)
    idx2, val2 = prec.make_sparse_vector(1000, 0.05, seed=3)
    assert idx1.size == val1.size == 50
    assert np.array_equal(idx1, idx2) and np.allclose(val1, val2)
    assert np.all(np.diff(idx1) > 0)  # sorted, unique


def test_spmspv_oracle_matches_dense_and_rejects_perturbation():
    a = _rand()
    idx, val = prec.make_sparse_vector(a.shape[0], 0.1, seed=1)
    y = prec.oracle_spmspv(a, idx, val)
    x = np.zeros(a.shape[0], np.float32)
    x[idx] = val
    assert np.allclose(y, a.toarray() @ x, rtol=1e-4, atol=1e-4)
    assert prec.validate_spmspv(y, a, idx, val, "fp32")
    assert not prec.validate_spmspv(y + 1.0, a, idx, val, "fp32")


def test_flops_spmspv_counts_touched_products():
    a = _rand()
    idx, _ = prec.make_sparse_vector(a.shape[0], 0.1, seed=2)
    col_nnz = np.diff(a.tocsc().indptr)
    assert flops_spmspv(int(col_nnz[idx].sum())) == 2 * int(col_nnz[idx].sum())


# --- cases ------------------------------------------------------------------------

def test_spmspv_case_from_matrix():
    a = _rand()
    c = cases.spmspv_case_from_matrix(a, case_id="t", graph_id="g")
    assert c.n == a.shape[0] and c.nnz == a.nnz
    assert c.vector_sparsities  # defaults populated
    assert cases.stratification(c)["density_bucket"]  # stratification handles the new case


# --- ingest canonical cells -------------------------------------------------------

def test_ingest_spmspv_cell_keyed_on_vector_sparsity():
    p = ingest._canonical_problem({"vector_sparsity": 0.05, "vector_nnz": 10}, "spmspv")
    assert p == {"vector_sparsity": 0.05}


def test_ingest_spgemm_cell_has_no_free_axis():
    assert ingest._canonical_problem({"N": 4}, "spgemm") == {}


def test_ingest_spmspv_target_passthrough():
    assert ingest._canonical_target("spmspv") == "spmspv"
    assert ingest._canonical_target("spmv") == "spmm"  # unchanged behavior
