"""Tests for the standard PyTorch providers in the RSNN comparison."""

from __future__ import annotations

import pytest
import torch

from benchmark.benchmark_persistent_snn import (
    BenchCase,
    make_input_sequence,
    make_recurrent_csr,
)
from benchmark.benchmark_rsnn_cudagraph_compare import (
    PROVIDERS,
    DirectCuSparseProvider,
    make_eager_runner,
    make_torch_csr_weight,
)
from btorch.sparse import CSR


def test_torch_csr_weight_uses_post_by_pre_orientation():
    """The public PyTorch weight should map source columns to target rows."""

    # btorch stores the edge 0 -> 2 in source-oriented CSR row 0. PyTorch
    # computes W @ z, so the converted weight must contain W[2, 0], not
    # W[0, 2]. This small asymmetric graph catches an accidental transpose.
    matrix = CSR.from_edges(
        row=torch.tensor([0]),
        col=torch.tensor([2]),
        data=torch.tensor([0.5]),
        shape=(3, 3),
    )

    weight = make_torch_csr_weight(matrix)

    assert weight.layout == torch.sparse_csr
    torch.testing.assert_close(
        weight.to_dense(),
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        ),
    )


def test_standard_dense_and_csr_rsnn_baselines_are_equivalent():
    """Dense linear and CSR sparse.mm should evaluate identical dynamics."""

    # Use multiple batches and timesteps so this checks both the B x N
    # transpose around sparse.mm and recurrent state propagation over time.
    case = BenchCase(
        n_neuron=32,
        batch_size=3,
        t_steps=8,
        fanout=5,
        event_rate=0.1,
    )
    device = torch.device("cpu")
    x_seq = make_input_sequence(case, device)
    matrix = make_recurrent_csr(case, device)
    csr_weight = make_torch_csr_weight(matrix)

    dense = make_eager_runner(
        x_seq, csr_weight.to_dense(), case, sparse=False
    )()
    sparse = make_eager_runner(x_seq, csr_weight, case, sparse=True)()

    torch.testing.assert_close(dense.spikes, sparse.spikes, atol=0, rtol=0)
    torch.testing.assert_close(dense.v, sparse.v, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dense.psc, sparse.psc, atol=1e-6, rtol=1e-6)


def test_default_providers_are_standard_pytorch_and_persistent_variants():
    """The published default comparison should not include custom scatter."""

    assert PROVIDERS == (
        "torch_dense_eager",
        "torch_dense_cudagraph",
        "torch_csr_eager",
        "torch_csr_cudagraph",
        "cusparse_direct_eager",
        "cusparse_direct_cudagraph",
        "persistent_plain",
        "persistent_binning",
        "persistent_spike_block",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("use_cudagraph", [False, True])
def test_direct_cusparse_matches_dense_baseline(batch_size, use_cudagraph):
    """Direct SpMV/SpMM and its graph capture should match dense PyTorch."""

    # Batch one exercises cuSPARSE SpMV; batch three exercises SpMM. Testing
    # both eager and captured execution also catches descriptors or workspace
    # whose lifetime ends before CUDA graph replay.
    case = BenchCase(
        n_neuron=64,
        batch_size=batch_size,
        t_steps=8,
        fanout=7,
        event_rate=0.1,
    )
    device = torch.device("cuda")
    x_seq = make_input_sequence(case, device)
    matrix = make_recurrent_csr(case, device)
    csr_weight = make_torch_csr_weight(matrix)
    reference = make_eager_runner(
        x_seq, csr_weight.to_dense(), case, sparse=False
    )()
    run = DirectCuSparseProvider().fixed_runner(
        x_seq,
        csr_weight,
        case,
        use_cudagraph=use_cudagraph,
    )

    result = run()
    torch.cuda.synchronize()

    torch.testing.assert_close(result.spikes, reference.spikes, atol=0, rtol=0)
    torch.testing.assert_close(result.v, reference.v, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result.psc, reference.psc, atol=1e-4, rtol=1e-4)
