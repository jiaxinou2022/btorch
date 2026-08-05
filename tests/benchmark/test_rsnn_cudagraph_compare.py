"""Tests for the standard PyTorch providers in the RSNN comparison."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import benchmark.benchmark_rsnn_cudagraph_compare as comparison
import benchmark.benchmark_rsnn_roofline as roofline
import benchmark.sota_rsnn_cudagraph as sota_adapters
from benchmark.benchmark_persistent_snn import (
    BenchCase,
    RSNNResult,
    make_input_sequence,
    make_recurrent_csr,
)
from benchmark.benchmark_rsnn_cudagraph_compare import (
    FLYBRAIN_DEFAULT_PROVIDERS,
    PROVIDERS,
    DirectCuSparseProvider,
    comparable_rows,
    correctness_metrics,
    latency_summary,
    load_flybrain_csr,
    make_eager_runner,
    make_torch_csr_weight,
    median_ms,
    pad_csr_neurons,
    provider_supports,
    resolve_dataset_defaults,
    time_samples_ms,
)
from benchmark.external_rsnn_simulators import (
    _genn_simulate,
    _patch_brian_timer,
    _source_indices,
    _summarize_log,
)
from benchmark.provider_common import BenchmarkRunner, PreparedMetadata
from benchmark.sota_rsnn_cudagraph import (
    CUDA_SPMSPV_CUDAGRAPH_PROVIDERS,
    CUDA_SPMSPV_PROVIDERS,
    SOTA_CUDAGRAPH_PROVIDERS,
    SOTA_EAGER_PROVIDERS,
    SPMSPV_EAGER_PROVIDERS,
    prepare_eager_matmul,
    prepare_matmul,
    prepare_sputnik_operator,
    prepare_vdha_dense_operator,
    rsnn_forward,
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

    dense = make_eager_runner(x_seq, csr_weight.to_dense(), case, sparse=False)()
    sparse = make_eager_runner(x_seq, csr_weight, case, sparse=True)()

    torch.testing.assert_close(dense.spikes, sparse.spikes, atol=0, rtol=0)
    torch.testing.assert_close(dense.v, sparse.v, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dense.psc, sparse.psc, atol=1e-6, rtol=1e-6)


def test_sota_rsnn_loop_uses_dynamic_recurrent_operator():
    """The SOTA adapter should consume each timestep's computed spikes."""

    case = BenchCase(
        n_neuron=16,
        batch_size=2,
        t_steps=6,
        fanout=4,
        event_rate=0.2,
    )
    device = torch.device("cpu")
    x_seq = make_input_sequence(case, device)
    matrix = make_recurrent_csr(case, device)
    weight = make_torch_csr_weight(matrix).to_dense()
    v0 = torch.zeros(case.batch_size, case.n_neuron)
    psc0 = torch.zeros_like(v0)

    result = rsnn_forward(
        x_seq,
        lambda spikes: torch.nn.functional.linear(spikes, weight),
        v0,
        psc0,
        case,
    )
    reference = make_eager_runner(x_seq, weight, case, sparse=False)()

    torch.testing.assert_close(result.spikes, reference.spikes)
    torch.testing.assert_close(result.v, reference.v)
    torch.testing.assert_close(result.psc, reference.psc)


def test_non_capturable_sota_wrappers_use_named_eager_fallbacks():
    """Host-controlled providers should never be mislabeled as CUDA Graphs."""

    assert SOTA_CUDAGRAPH_PROVIDERS == (
        "sputnik_cudagraph",
        *CUDA_SPMSPV_CUDAGRAPH_PROVIDERS,
    )
    assert set(CUDA_SPMSPV_CUDAGRAPH_PROVIDERS.values()) == {
        "vdha",
        "vdha_pipe",
        "tilespmspv",
        "sortspmspv",
        "globalatomic",
        "blockatomic",
        "blocksort",
        "naivespmspv",
        "holaspmspv",
    }
    assert SOTA_EAGER_PROVIDERS[:4] == (
        "mh_spgemm_eager",
        "dtc_spmm_eager",
        "flashsparse_eager",
        "torch_csr_host_e2e",
    )
    assert SOTA_EAGER_PROVIDERS[4:] == SPMSPV_EAGER_PROVIDERS
    assert set(CUDA_SPMSPV_PROVIDERS.values()) == {
        "vdha",
        "vdha_pipe",
        "tilespmspv",
        "sortspmspv",
        "globalatomic",
        "blockatomic",
        "blocksort",
        "naivespmspv",
        "holaspmspv",
    }


def test_sputnik_batch_one_adapter_reuses_zero_copy_buffers(monkeypatch):
    """Sputnik B=1 should view the input and reuse one prepared output."""

    calls = []

    class FakeSputnikLibrary:
        def cbn_sputnik_prepare(self, *args):
            return 1

        def cbn_sputnik_compute_device(self, handle, rhs, output, stream):
            calls.append((rhs.value, output.value))
            return 0

        def cbn_sputnik_free(self, handle):
            return None

    fake_module = types.SimpleNamespace(_load_lib=lambda: FakeSputnikLibrary())
    monkeypatch.setitem(sys.modules, "connectome_bench_sputnik", fake_module)
    monkeypatch.setattr(sota_adapters, "_stream_pointer", lambda: None)
    weight = torch.eye(4).to_sparse_csr()
    case = BenchCase(
        n_neuron=4,
        batch_size=1,
        t_steps=1,
        fanout=1,
        event_rate=0.1,
    )
    matmul, release = prepare_matmul("sputnik_cudagraph", weight, case)
    spikes = torch.ones(1, 4)

    first = matmul(spikes)
    second = matmul(spikes)
    release()

    assert getattr(matmul, "_btorch_transform_mode") == "zero_copy_view"
    assert first.data_ptr() == second.data_ptr()
    assert calls[0][0] == spikes.data_ptr()
    assert calls[1][0] == spikes.data_ptr()
    assert calls[0][1] == calls[1][1] == first.data_ptr()


def test_vdha_graph_adapter_reuses_preallocated_output(monkeypatch):
    """In-tree VDHA graph calls should reuse one prepared device output."""

    from connectome_dataset.benchmarks.cuda import kernels

    output_pointers = []

    class FakePreparedKernel:
        def __init__(self, spec, matrix, *, precision):
            assert spec.provider == "vdha"
            assert matrix.shape == (4, 4)
            assert precision == "fp32"

        def step_device(self, x_pointer, output_pointer, stream_pointer):
            output_pointers.append(output_pointer)

        def free(self):
            return None

    monkeypatch.setattr(kernels, "PreparedKernel", FakePreparedKernel)
    monkeypatch.setattr(kernels, "supports_device_compute", lambda spec, p: True)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(cuda_stream=123),
    )
    weight = torch.eye(4).to_sparse_csr()
    case = BenchCase(
        n_neuron=4,
        batch_size=1,
        t_steps=1,
        fanout=1,
        event_rate=0.1,
    )
    matmul, release = prepare_matmul("vdha_cudagraph", weight, case)
    spikes = torch.ones(1, 4)

    first = matmul(spikes)
    second = matmul(spikes)
    release()

    assert first.data_ptr() == second.data_ptr()
    assert output_pointers == [first.data_ptr(), first.data_ptr()]


def test_structured_sputnik_operator_has_distinct_native_entry(monkeypatch):
    """Native NB and adapted BN traces should use separate output buffers."""

    calls = []

    class FakeSputnikLibrary:
        def cbn_sputnik_prepare(self, *args):
            return 1

        def cbn_sputnik_compute_device(self, handle, rhs, output, stream):
            calls.append((rhs.value, output.value))
            return 0

        def cbn_sputnik_free(self, handle):
            return None

    monkeypatch.setitem(
        sys.modules,
        "connectome_bench_sputnik",
        types.SimpleNamespace(_load_lib=lambda: FakeSputnikLibrary()),
    )
    monkeypatch.setattr(sota_adapters, "_stream_pointer", lambda: None)
    weight = torch.eye(4).to_sparse_csr()
    case = BenchCase(
        n_neuron=4,
        batch_size=1,
        t_steps=2,
        fanout=1,
        event_rate=0.1,
    )
    spikes = torch.ones(2, 1, 4)

    prepared = prepare_sputnik_operator(weight, spikes, case)
    prepared.native_run()
    prepared.adapted_run()

    assert prepared.metadata.native_available
    assert prepared.transform_run is None
    assert prepared.native_output is not None
    assert prepared.native_output.data_ptr() != prepared.adapted_output.data_ptr()
    assert len(calls) == 4
    prepared.release()


def test_structured_vdha_dense_does_not_invent_native_curve(monkeypatch):
    """Dense VDHA has one valid dynamic path until a sparse device ABI exists."""

    class FakeVdhaLibrary:
        def cbn_vdha_prepare_dense(self, *args):
            return 1

        def cbn_vdha_compute_dense_device(self, *args):
            return 0

        def cbn_vdha_free(self, handle):
            return None

    monkeypatch.setitem(
        sys.modules,
        "connectome_bench_vdha",
        types.SimpleNamespace(_load_lib=lambda dtype: FakeVdhaLibrary()),
    )
    monkeypatch.setattr(sota_adapters, "_stream_pointer", lambda: None)
    weight = torch.eye(4).to_sparse_csr()
    case = BenchCase(
        n_neuron=4,
        batch_size=1,
        t_steps=2,
        fanout=1,
        event_rate=0.1,
    )

    prepared = prepare_vdha_dense_operator(
        weight,
        torch.ones(2, 1, 4),
        case,
    )

    assert prepared.native_run is None
    assert prepared.native_output is None
    assert not prepared.metadata.native_available
    assert prepared.metadata.native_status == "not_distinct_from_adapter"
    prepared.release()


def test_split_spmspv_adapter_reuses_matrix_and_accepts_empty_spikes(
    monkeypatch,
):
    """The RSNN adapter should preprocess A once and update only sparse x."""

    from connectome_dataset.benchmarks.cuda import kernels

    calls = {"prepare": 0, "step": [], "free": 0}

    class FakePreparedKernel:
        def __init__(self, spec, matrix, *, precision):
            calls["prepare"] += 1
            assert spec.provider == "vdha"
            assert precision == "fp32"
            assert matrix.shape == (4, 4)

        def step(self, indices, values):
            calls["step"].append((indices.copy(), values.copy()))
            output = np.zeros(4, dtype=np.float32)
            output[indices] = values
            return output

        def free(self):
            calls["free"] += 1

    monkeypatch.setattr(kernels, "PreparedKernel", FakePreparedKernel)
    monkeypatch.setattr(kernels, "supports_preprocess", lambda spec, p: True)
    weight = torch.eye(4).to_sparse_csr()
    case = BenchCase(
        n_neuron=4,
        batch_size=1,
        t_steps=2,
        fanout=1,
        event_rate=0.1,
    )

    matmul = prepare_eager_matmul("vdha_spmspv_eager", weight, case)
    nonempty = matmul(torch.tensor([[0.0, 1.0, 0.0, 1.0]]))
    empty = matmul(torch.zeros(1, 4))
    getattr(matmul, "_btorch_release")()

    torch.testing.assert_close(nonempty, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))
    torch.testing.assert_close(empty, torch.zeros(1, 4))
    assert calls["prepare"] == 1
    assert calls["step"][0][0].tolist() == [1, 3]
    assert calls["step"][1][0].size == 0
    assert calls["free"] == 1


def test_split_spmspv_graph_adapter_uses_stable_device_pointers(monkeypatch):
    """Graph adapters should pass only stable tensors and the current stream."""

    from connectome_dataset.benchmarks.cuda import kernels

    calls = []

    class FakePreparedKernel:
        def __init__(self, spec, matrix, *, precision):
            assert spec.provider == "tilespmspv"
            assert matrix.shape == (4, 4)
            assert precision == "fp32"

        def step_device(self, x_pointer, output_pointer, stream_pointer):
            calls.append((x_pointer, output_pointer, stream_pointer))

        def free(self):
            return None

    monkeypatch.setattr(kernels, "PreparedKernel", FakePreparedKernel)
    monkeypatch.setattr(kernels, "supports_device_compute", lambda spec, p: True)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(cuda_stream=123),
    )
    weight = torch.eye(4).to_sparse_csr()
    case = BenchCase(
        n_neuron=4,
        batch_size=1,
        t_steps=2,
        fanout=1,
        event_rate=0.1,
    )
    matmul, release = prepare_matmul("tilespmspv_cudagraph", weight, case)
    spikes = torch.ones(1, 4)

    first = matmul(spikes)
    second = matmul(spikes)
    release()

    assert first.data_ptr() == second.data_ptr()
    assert calls == [
        (spikes.data_ptr(), first.data_ptr(), 123),
        (spikes.data_ptr(), first.data_ptr(), 123),
    ]
    assert getattr(matmul, "_btorch_spike_representation") == "dense_device"


def test_adaptive_spmspv_freezes_lazy_handles_after_priming(monkeypatch, tmp_path):
    """Timed selector execution must not preprocess a newly selected kernel."""

    from connectome_dataset.benchmarks.cuda import adaptive, kernels

    class FakeSelector:
        def predict(self, features):
            return "globalatomic" if features["nnz_x"] <= 1 else "naivespmspv"

    class FakePreparedKernel:
        freed = []

        def __init__(self, spec, matrix, *, precision):
            self.provider = spec.provider

        def step(self, indices, values):
            return np.zeros(4, dtype=np.float32)

        def free(self):
            self.freed.append(self.provider)

    monkeypatch.setattr(
        adaptive.AdaptiveSelector,
        "load",
        classmethod(lambda cls, path: FakeSelector()),
    )
    monkeypatch.setattr(kernels, "PreparedKernel", FakePreparedKernel)
    monkeypatch.setattr(kernels, "supports_preprocess", lambda spec, p: True)
    model = tmp_path / "selector.joblib"
    model.write_bytes(b"fake")
    weight = torch.eye(4).to_sparse_csr()
    case = BenchCase(
        n_neuron=4,
        batch_size=1,
        t_steps=1,
        fanout=1,
        event_rate=0.1,
    )
    matmul = prepare_eager_matmul(
        "adaptive_spmspv_eager",
        weight,
        case,
        selector_model_paths={"adaptive_spmspv_eager": model},
    )

    matmul(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    getattr(matmul, "_btorch_freeze_after_prime")()
    matmul(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    with pytest.raises(RuntimeError, match="unprimed kernel"):
        matmul(torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
    getattr(matmul, "_btorch_release")()

    assert FakePreparedKernel.freed == ["globalatomic"]


@pytest.mark.parametrize(
    ("provider", "module_name"),
    [
        (
            "dtc_spmm_eager",
            "connectome_dataset.benchmarks.torch.dtc.spmm",
        ),
        (
            "flashsparse_eager",
            "connectome_dataset.benchmarks.torch.flashsparse.spmm",
        ),
    ],
)
def test_dtc_and_flashsparse_preprocess_once_per_runner(
    monkeypatch, provider, module_name
):
    """A timestep must reuse one prepared sparse plan."""

    module = __import__(module_name, fromlist=["spmm"])
    calls = {"prepare": 0, "run": 0}

    def fake_make_dynamic_fn(matrix, **kwargs):
        calls["prepare"] += 1
        assert matrix.nnz == 4
        assert kwargs["batch_size"] == 32

        def dynamic_run(rhs):
            calls["run"] += 1
            return rhs.clone(), 0.0

        return dynamic_run

    monkeypatch.setattr(module, "make_dynamic_fn", fake_make_dynamic_fn, raising=False)
    row = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    col = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    weight = torch.sparse_csr_tensor(
        row,
        col,
        torch.ones(4),
        size=(4, 4),
    )
    case = BenchCase(
        n_neuron=4,
        batch_size=32,
        t_steps=5,
        fanout=1,
        event_rate=0.1,
    )

    matmul = prepare_eager_matmul(provider, weight, case)
    for _ in range(case.t_steps):
        output = matmul(torch.ones(case.batch_size, case.n_neuron))

    assert calls == {"prepare": 1, "run": case.t_steps}
    assert output.shape == (case.batch_size, case.n_neuron)


def test_sota_capabilities_reject_unsupported_batch_one_providers():
    """Fixed-width providers should be skipped without changing the workload."""

    case = BenchCase(
        n_neuron=4096,
        batch_size=1,
        t_steps=32,
        fanout=32,
        event_rate=0.01,
    )

    for provider in (
        "torch_csr_cudagraph",
        "sputnik_cudagraph",
        *CUDA_SPMSPV_CUDAGRAPH_PROVIDERS,
        "torch_csr_host_e2e",
        *SPMSPV_EAGER_PROVIDERS,
    ):
        assert provider_supports(provider, case) == (True, "")

    for provider in (
        "mh_spgemm_eager",
        "dtc_spmm_eager",
        "flashsparse_eager",
    ):
        supported, reason = provider_supports(provider, case)
        assert not supported
        assert "batch_size=1" in reason


def test_csr_neuron_padding_adds_only_isolated_rows():
    """Neuron padding should preserve every edge and avoid batch padding."""

    matrix = CSR.from_edges(
        row=torch.tensor([0, 2]),
        col=torch.tensor([1, 0]),
        data=torch.tensor([0.5, -0.25]),
        shape=(3, 3),
    )

    padded, info = pad_csr_neurons(matrix, n_alignment=4)

    assert info.logical_n == 3
    assert info.physical_n == 4
    assert info.padding_ratio == pytest.approx(4 / 3)
    assert padded.shape == (4, 4)
    assert padded.indptr.tolist() == [0, 1, 1, 2, 2]
    torch.testing.assert_close(padded.indices, matrix.indices)
    torch.testing.assert_close(padded.data, matrix.data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_before_sample_reset_policy_resets_every_timed_inference():
    """Stateful timing must reset outside every measured CUDA interval."""

    state = torch.ones(1, device="cuda")
    resets = 0

    def reset():
        nonlocal resets
        resets += 1
        state.zero_()

    def run():
        state.add_(1)
        return RSNNResult(
            spikes=state.view(1, 1, 1),
            v=state.view(1, 1),
            psc=state.view(1, 1),
        )

    metadata = PreparedMetadata(
        logical_shape=(1, 1, 1),
        physical_shape=(1, 1, 1),
        input_layout="BN",
        operator_layout="BN",
        index_dtype="none",
        value_dtype="torch.float32",
        compute_dtype="torch.float32",
        workspace_bytes=0,
        persistent_bytes=state.numel() * state.element_size(),
        padding_ratio=1.0,
        execution_class="device_native",
        timing_scope="gpu_execution",
    )
    runner = BenchmarkRunner(
        run_fn=run,
        reset_fn=reset,
        metadata=metadata,
        reset_policy="before_sample",
    )

    samples = time_samples_ms(runner, warmup=2, repeat=3)

    assert len(samples) == 3
    # One reset starts each phase, and every warmup/timed sample gets its own
    # reset. Those reset kernels are enqueued before the CUDA start event.
    assert resets == 2 + 2 + 3
    torch.testing.assert_close(state, torch.ones_like(state))


def test_comparison_group_isolates_adapted_and_host_execution():
    """Speedups must not cross execution-class or timing-scope boundaries."""

    baseline = {
        "dataset": "uniform",
        "logical_n": 4096,
        "logical_batch": 1,
        "t_steps": 32,
        "timing_scope": "gpu_execution",
        "execution_class": "device_native",
        "benchmark_mode": "full_rsnn",
        "state_semantics": "independent_inference",
    }

    assert comparable_rows(dict(baseline), baseline)
    for field, value in (
        ("execution_class", "device_adapted"),
        ("timing_scope", "public_wrapper_e2e"),
        ("logical_batch", 32),
        ("state_semantics", "continuous_samples"),
    ):
        candidate = dict(baseline)
        candidate[field] = value
        assert not comparable_rows(candidate, baseline)


def test_latency_median_averages_two_middle_samples():
    """Even-sized benchmark samples should use the statistical median."""

    assert median_ms([10.0, 1.0, 4.0, 2.0]) == pytest.approx(3.0)


def test_latency_summary_reports_sample_standard_deviation():
    """CSV summaries should retain conventional sample statistics."""

    summary = latency_summary([1.0, 2.0, 3.0])

    assert summary == {
        "sample_count": 3,
        "latency_mean_ms": pytest.approx(2.0),
        "latency_std_ms": pytest.approx(1.0),
        "latency_min_ms": pytest.approx(1.0),
        "latency_max_ms": pytest.approx(3.0),
    }


def test_default_providers_include_external_simulator_comparisons():
    """The published comparison should include both generated-code simulators."""

    assert PROVIDERS == (
        "torch_dense_eager",
        "torch_dense_cudagraph",
        "torch_csr_eager",
        "torch_csr_cudagraph",
        "cusparse_direct_eager",
        "cusparse_direct_cudagraph",
        "sputnik_cudagraph",
        *CUDA_SPMSPV_CUDAGRAPH_PROVIDERS,
        "mh_spgemm_eager",
        "dtc_spmm_eager",
        "flashsparse_eager",
        "torch_csr_host_e2e",
        *SPMSPV_EAGER_PROVIDERS,
        "persistent_plain",
        "persistent_binning",
        "persistent_spike_block",
        "genn",
        "brian2cuda",
    )


def test_flybrain_defaults_avoid_dense_whole_brain_weights():
    """FlyBrain defaults should remain sparse at whole-connectome scale.

    FlyWire contains enough neurons that a dense adjacency commonly exceeds
    accelerator memory. Keeping both dense providers out of this dataset's
    implicit provider list prevents an unexpected allocation while leaving
    them available through an explicit ``--providers`` argument.
    """

    assert FLYBRAIN_DEFAULT_PROVIDERS
    assert all(
        not provider.startswith("torch_dense_")
        for provider in FLYBRAIN_DEFAULT_PROVIDERS
    )
    assert set(FLYBRAIN_DEFAULT_PROVIDERS).issubset(PROVIDERS)


def test_flybrain_loader_preserves_signed_weights_and_orientation(monkeypatch):
    """FlyBrain loading should scale signed pre-to-post synapse counts.

    The asymmetric edge locations verify that the loader does not transpose
    the source-oriented CSR graph. Positive and negative values verify that
    excitatory and inhibitory FlyWire connections both survive conversion.
    """

    source = sp.csr_matrix(
        (
            [2.0, -3.0],
            ([0, 2], [1, 0]),
        ),
        shape=(3, 3),
        dtype="float32",
    )
    calls = {}

    def fake_load_flywire_783(*, root, use_weights):
        calls.update(root=root, use_weights=use_weights)
        return source.copy()

    monkeypatch.setattr(
        "connectome_dataset.graph_loader.load_flywire_783",
        fake_load_flywire_783,
    )
    matrix = load_flybrain_csr(
        None,
        weight_scale=0.25,
        device=torch.device("cpu"),
    )

    assert calls == {"root": None, "use_weights": True}
    torch.testing.assert_close(
        matrix.to_dense(),
        torch.tensor(
            [
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [-0.75, 0.0, 0.0],
            ]
        ),
    )


def test_flybrain_is_the_default_dataset(monkeypatch):
    """Both command-line benchmarks should select FlyBrain by default."""

    monkeypatch.setattr(sys, "argv", [comparison.__file__])
    comparison_args = comparison.parse_args()
    monkeypatch.setattr(sys, "argv", [roofline.__file__])
    roofline_args = roofline.parse_args()

    assert comparison_args.dataset == "flybrain"
    assert comparison_args.weight_scale == pytest.approx(0.275)
    assert comparison_args.eager_warmup == 0
    assert comparison_args.eager_repeat == 1
    assert not comparison_args.continuous_state
    assert roofline_args.dataset == "flybrain"
    assert roofline_args.weight_scale == pytest.approx(0.275)


def test_external_framework_tuning_options_are_explicit(monkeypatch):
    """Framework tuning choices should be reproducible from the CLI."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            comparison.__file__,
            "--no-genn-optimize-code",
            "--brian2cuda-sm-multiplier",
            "2",
            "--brian2cuda-parallel-blocks",
            "0",
            "--no-brian2cuda-calc-occupancy",
            "--brian2cuda-syn-launch-bounds",
        ],
    )

    args = comparison.parse_args()

    assert not args.genn_optimize_code
    assert args.brian2cuda_sm_multiplier == 2
    assert args.brian2cuda_parallel_blocks == 0
    assert not args.brian2cuda_calc_occupancy
    assert args.brian2cuda_syn_launch_bounds


def test_dataset_specific_weight_scale_defaults():
    """Dataset defaults should preserve legacy mouse and uniform behavior."""

    assert resolve_dataset_defaults("flybrain", None) == ("flybrain", 0.275)
    assert resolve_dataset_defaults("flywire_783", None) == ("flybrain", 0.275)
    assert resolve_dataset_defaults("mice_column_v1", None) == (
        "mice_column_v1",
        0.15,
    )
    assert resolve_dataset_defaults("uniform", None) == ("uniform", 0.15)
    assert resolve_dataset_defaults("flybrain", 0.5) == ("flybrain", 0.5)


def test_external_csr_source_expansion_preserves_empty_rows():
    """External simulators should receive the exact pre-to-post edge list.

    The empty middle row makes this more than a simple fixed-fanout example
    and checks the conversion used for both GeNN and Brian2CUDA.
    """

    source = _source_indices(np.array([0, 2, 2, 3], dtype=np.int64))

    np.testing.assert_array_equal(source, np.array([0, 0, 2]))


def test_brian_timer_patch_synchronizes_both_interval_boundaries(tmp_path: Path):
    """Brian2CUDA timing should exclude preceding work and include queued work."""

    template = (
        "void Network::run() {\n"
        "    start = std::chrono::high_resolution_clock::now();\n"
        "    Network::_globally_running = false;\n\n"
        "    current = std::chrono::high_resolution_clock::now();\n"
        "}\n"
    )
    path = tmp_path / "network.cu"
    path.write_text(template)

    _patch_brian_timer(tmp_path)

    patched = path.read_text()
    assert patched.count("CUDA_SAFE_CALL(cudaDeviceSynchronize());") == 2
    assert patched.index("cudaDeviceSynchronize") < patched.index("start =")
    assert patched.rindex("cudaDeviceSynchronize") < patched.index("current =")


def test_external_error_log_summary_is_bounded_and_keeps_both_ends():
    """Large compiler failures should not flood or stall benchmark output."""

    output = "diagnostic-start\n" + ("x" * 100_000) + "\ndiagnostic-end"

    summary = _summarize_log(output, max_chars=1_000)

    assert "diagnostic-start" in summary
    assert "diagnostic-end" in summary
    assert "omitted" in summary
    assert len(summary) < 1_100


def test_genn_timing_uses_native_accumulated_cuda_events():
    """GeNN timing should use its runtime's events rather than PyTorch CUDA.

    GeNN loads a separate CUDA runtime inside its worker. This fake model
    checks that a sample is the delta of all four documented kernel timers,
    which also provides the required GeNN-side completion barrier.
    """

    class FakeGeNNModel:
        neuron_update_time = 0.0
        presynaptic_update_time = 0.0
        postsynaptic_update_time = 0.0
        synapse_dynamics_time = 0.0

        def step_time(self):
            self.neuron_update_time += 1e-3
            self.presynaptic_update_time += 2e-3
            self.postsynaptic_update_time += 3e-3
            self.synapse_dynamics_time += 4e-3

    case = BenchCase(
        n_neuron=4,
        batch_size=1,
        t_steps=5,
        fanout=1,
        event_rate=0.1,
    )

    assert _genn_simulate(FakeGeNNModel(), case) == pytest.approx(50.0)


def test_correctness_accepts_small_relative_error_on_large_states():
    """Large FlyWire states should not fail solely due to absolute error."""

    reference = RSNNResult(
        spikes=torch.zeros(2, 1, 2),
        v=torch.tensor([[190_525.0, -80_000.0]]),
        psc=torch.tensor([[66_377.0, -40_000.0]]),
    )
    result = RSNNResult(
        spikes=reference.spikes.clone(),
        v=reference.v + torch.tensor([[0.72, -0.30]]),
        psc=reference.psc + torch.tensor([[0.48, -0.20]]),
    )

    metrics = correctness_metrics(result, reference)

    assert metrics["status"] == "passed"
    assert metrics["v_max_abs_diff"] > 0.2
    assert metrics["psc_max_abs_diff"] > 0.005
    assert metrics["v_max_normalized_error"] < 1.0
    assert metrics["psc_max_normalized_error"] < 1.0


def test_correctness_rejects_material_error_near_zero():
    """Relative tolerance should not hide incorrect low-magnitude state."""

    reference = RSNNResult(
        spikes=torch.zeros(1, 1, 1),
        v=torch.zeros(1, 1),
        psc=torch.zeros(1, 1),
    )
    result = RSNNResult(
        spikes=reference.spikes.clone(),
        v=torch.tensor([[0.21]]),
        psc=torch.tensor([[0.006]]),
    )

    metrics = correctness_metrics(result, reference)

    assert metrics["status"] == "correctness_failed"


def test_spike_equivalence_retains_state_errors_as_diagnostics():
    """Approximate recurrent kernels should be judged by spike trajectories."""

    reference = RSNNResult(
        spikes=torch.zeros(2, 1, 2),
        v=torch.zeros(1, 2),
        psc=torch.zeros(1, 2),
    )
    result = RSNNResult(
        spikes=reference.spikes.clone(),
        v=torch.tensor([[10.0, -10.0]]),
        psc=torch.tensor([[5.0, -5.0]]),
    )

    metrics = correctness_metrics(result, reference, strict_state=False)

    assert metrics["status"] == "passed_spike_equivalent"
    assert metrics["v_max_normalized_error"] > 1.0
    assert metrics["psc_max_normalized_error"] > 1.0


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
    reference = make_eager_runner(x_seq, csr_weight.to_dense(), case, sparse=False)()
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
