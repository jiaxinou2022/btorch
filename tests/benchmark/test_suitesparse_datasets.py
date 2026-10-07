"""Tests for reproducible SuiteSparse benchmark dataset loading."""

from __future__ import annotations

import io
import tarfile

import scipy.sparse
import torch

from benchmark.prepare_suitesparse_dataset import find_matrix_member
from benchmark.suitesparse_datasets import (
    SUITESPARSE_DATASETS,
    SuiteSparseDataset,
    load_suitesparse_csr,
    resolve_prepared_path,
)


def test_resolve_prepared_path_accepts_directory(tmp_path) -> None:
    """Resolve the canonical benchmark name inside a processed directory."""

    path = tmp_path / "orkut.npz"
    path.touch()

    assert resolve_prepared_path(tmp_path, "orkut") == path


def test_find_matrix_member_ignores_auxiliary_matrices(tmp_path) -> None:
    """Select the primary graph from archives that also contain metadata."""

    archive_path = tmp_path / "graph.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name in ("com-Orkut/com-Orkut.mtx", "com-Orkut/Communities.mtx"):
            payload = b"%%MatrixMarket matrix coordinate real general\n"
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    with tarfile.open(archive_path) as archive:
        member = find_matrix_member(archive, "com-Orkut")

    assert member.name == "com-Orkut/com-Orkut.mtx"


def test_load_binary_matrix_normalizes_mean_fanout(tmp_path, monkeypatch) -> None:
    """Give every binary edge equal strength with a controlled row-sum mean."""

    dataset = SuiteSparseDataset("Test", "tiny", 3, 3, "binary")
    monkeypatch.setitem(SUITESPARSE_DATASETS, "tiny_binary", dataset)
    matrix = scipy.sparse.csr_matrix(
        (
            torch.ones(3).numpy(),
            (torch.tensor([0, 0, 2]).numpy(), torch.tensor([1, 2, 0]).numpy()),
        ),
        shape=(3, 3),
    )
    scipy.sparse.save_npz(tmp_path / "tiny_binary.npz", matrix)
    scipy.sparse.save_npz(
        tmp_path / "tiny_binary_transpose.npz",
        matrix.transpose().tocsr(),
    )

    loaded = load_suitesparse_csr(
        "tiny_binary",
        tmp_path,
        weight_scale=0.15,
        device=torch.device("cpu"),
    )

    assert loaded.shape == (3, 3)
    assert loaded.indices.numel() == 3
    assert torch.allclose(loaded.data, torch.full((3,), 0.15))
    assert "torch_csr_weight" in loaded._cache


def test_load_real_matrix_preserves_relative_signed_values(
    tmp_path,
    monkeypatch,
) -> None:
    """Retain signs and ratios while normalizing aggregate absolute weight."""

    dataset = SuiteSparseDataset("Test", "tiny", 2, 2, "real")
    monkeypatch.setitem(SUITESPARSE_DATASETS, "tiny_real", dataset)
    matrix = scipy.sparse.csr_matrix(
        (
            torch.tensor([2.0, -1.0]).numpy(),
            (torch.tensor([0, 1]).numpy(), torch.tensor([1, 0]).numpy()),
        ),
        shape=(2, 2),
    )
    scipy.sparse.save_npz(tmp_path / "tiny_real.npz", matrix)
    scipy.sparse.save_npz(
        tmp_path / "tiny_real_transpose.npz",
        matrix.transpose().tocsr(),
    )

    loaded = load_suitesparse_csr(
        "tiny_real",
        tmp_path,
        weight_scale=0.15,
        device=torch.device("cpu"),
    )

    assert torch.allclose(loaded.data, torch.tensor([0.2, -0.1]))
    assert torch.isclose(loaded.data.abs().sum() / 2, torch.tensor(0.15))
