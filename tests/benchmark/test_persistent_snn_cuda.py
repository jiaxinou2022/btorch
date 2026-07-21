import math

import pytest
import torch

from btorch.backend.persistent_snn import (
    EventCSRGraph,
    PersistentSNNParams,
    PersistentSNNState,
    WindowedSpikeEvents,
    build_persistent_snn_reorder_plan,
    make_persistent_snn_workspace,
    persistent_snn_forward,
    reorder_persistent_snn_state,
    reorder_windowed_spike_events,
    restore_persistent_snn_output,
)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return torch.device("cuda")


def _run_cuda_or_skip(*args, **kwargs):
    try:
        return persistent_snn_forward(*args, backend="cuda_persistent", **kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        skippable = (
            "cuda_home",
            "nvcc",
            "ninja",
            "cooperative",
            "no kernel image",
            "unsupported",
        )
        if any(token in message for token in skippable):
            pytest.skip(f"CUDA persistent backend unavailable: {exc}")
        raise


def _dense_to_events(x_seq: torch.Tensor) -> WindowedSpikeEvents:
    """Convert dense external currents to the CUDA event contract."""

    t_steps, batch_size, n_neuron = x_seq.shape
    active = x_seq != 0
    counts = active.sum(dim=2, dtype=torch.int32).reshape(-1)
    offsets = torch.zeros(counts.numel() + 1, device=x_seq.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0)
    indices = torch.nonzero(active.reshape(-1, n_neuron), as_tuple=False)[:, 1]
    return WindowedSpikeEvents(
        offsets=offsets.contiguous(),
        indices=indices.to(torch.int32).contiguous(),
        values=x_seq[active].to(torch.float32).contiguous(),
        shape=(t_steps, batch_size, n_neuron),
    )


def _graph(device: torch.device) -> tuple[EventCSRGraph, torch.Tensor]:
    """Create a small recurrent graph with varied fanout for correctness tests."""

    n_neuron = 4
    edges = [
        (0, 1, 0.25),
        (0, 2, 0.50),
        (1, 2, 0.75),
        (2, 0, -0.25),
        (2, 3, 0.40),
        (3, 1, 0.10),
    ]
    indptr = torch.zeros(n_neuron + 1, device=device, dtype=torch.int32)
    for pre, _post, _weight in edges:
        indptr[pre + 1] += 1
    indptr = torch.cumsum(indptr, dim=0).to(torch.int32)
    indices = torch.tensor([post for _pre, post, _weight in edges], device=device)
    indices = indices.to(torch.int32)
    weight = torch.tensor([weight for _pre, _post, weight in edges], device=device)
    weight = weight.to(torch.float32)
    dense = torch.zeros(n_neuron, n_neuron, device=device)
    for pre, post, edge_weight in edges:
        dense[pre, post] = edge_weight
    return (
        EventCSRGraph(
            indptr=indptr.contiguous(),
            indices=indices.contiguous(),
            weight=weight.contiguous(),
            delay=None,
            shape=(n_neuron, n_neuron),
        ),
        dense,
    )


def _fanout_bucket_graph(
    device: torch.device,
    fanouts: tuple[int, ...],
) -> tuple[EventCSRGraph, torch.Tensor]:
    """Create a recurrent graph that exercises selected fanout buckets.

    The rows under test use unique post indices, so the dense reference has
    the same accumulation semantics as CSR traversal. This makes the test a
    small executable example for the 256-edge fanout boundary.
    """

    n_neuron = max(max(fanouts), len(fanouts)) + 1
    indptr_values = [0]
    all_indices = []
    all_weights = []
    dense = torch.zeros(n_neuron, n_neuron, device=device)
    for pre, fanout in enumerate(fanouts):
        posts = torch.arange(fanout, device=device, dtype=torch.int32)
        weight = torch.full(
            (fanout,),
            0.001 * (pre + 1),
            device=device,
            dtype=torch.float32,
        )
        all_indices.append(posts)
        all_weights.append(weight)
        dense[pre, posts.to(torch.long)] = weight
        indptr_values.append(indptr_values[-1] + fanout)
    for _ in range(len(fanouts), n_neuron):
        indptr_values.append(indptr_values[-1])

    graph = EventCSRGraph(
        indptr=torch.tensor(indptr_values, device=device, dtype=torch.int32),
        indices=torch.cat(all_indices).contiguous(),
        weight=torch.cat(all_weights).contiguous(),
        delay=None,
        shape=(n_neuron, n_neuron),
    )
    return graph, dense


def _reference(
    x_seq: torch.Tensor,
    weight_dense: torch.Tensor,
    state: PersistentSNNState,
    params: PersistentSNNParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense reference matching the v1 persistent CUDA contract."""

    v = state.v.clone()
    psc = state.psc.clone()
    decay = math.exp(-params.dt / params.tau_syn)
    spikes = []
    for t in range(x_seq.shape[0]):
        current = psc + x_seq[t]
        v_pre = v + params.dt * (
            -(v - params.v_reset) / params.tau_mem + current / params.c_m
        )
        z = (v_pre >= params.v_threshold).to(v.dtype)
        v = v_pre - (params.v_threshold - params.v_reset) * z
        psc = psc * decay + z @ weight_dense
        spikes.append(z)
    return torch.stack(spikes, dim=0), v, psc


def test_cuda_persistent_matches_dense_reference():
    """CUDA persistent dynamics should match dense RSNN reference."""

    device = _require_cuda()
    x_seq = torch.tensor(
        [
            [[1.2, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0]],
            [[0.0, 0.0, 1.4, 0.0], [0.0, 0.0, 0.0, 1.6]],
            [[0.6, 0.0, 0.7, 0.0], [0.8, 0.0, 0.0, 0.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    graph, dense = _graph(device)
    state = PersistentSNNState(
        v=torch.zeros(2, 4, device=device),
        psc=torch.zeros(2, 4, device=device),
    )
    params = PersistentSNNParams(tau_mem=20.0, tau_syn=5.0, window_size=3)

    out = _run_cuda_or_skip(
        _dense_to_events(x_seq),
        graph,
        state,
        params,
        return_mode="both",
    )
    ref_spikes, ref_v, ref_psc = _reference(x_seq, dense, state, params)

    assert out.spikes is not None
    assert out.spike_events is not None
    torch.testing.assert_close(out.spikes, ref_spikes, atol=0, rtol=0)
    torch.testing.assert_close(out.state.v, ref_v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.state.psc, ref_psc, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("return_mode", ["events", "both"])
def test_cuda_persistent_event_output_matches_dense_spikes(return_mode):
    """Event-only and combined modes should return the same fired indices.

    Testing event-only separately is important because its CUDA specialization
    does not allocate or write the dense ``T * B * N`` spike history.
    """

    device = _require_cuda()
    x_seq = torch.tensor(
        [[[1.2, 1.3, 0.0, 0.0]], [[0.0, 0.0, 1.4, 1.5]]],
        device=device,
        dtype=torch.float32,
    )
    graph, dense = _graph(device)
    state = PersistentSNNState(
        v=torch.zeros(1, 4, device=device),
        psc=torch.zeros(1, 4, device=device),
    )
    out = _run_cuda_or_skip(
        _dense_to_events(x_seq),
        graph,
        state,
        PersistentSNNParams(window_size=2),
        return_mode=return_mode,
    )

    expected_spikes, _expected_v, _expected_psc = _reference(
        x_seq,
        dense,
        state,
        PersistentSNNParams(window_size=2),
    )
    if return_mode == "events":
        assert out.spikes is None
    else:
        assert out.spikes is not None
        torch.testing.assert_close(out.spikes, expected_spikes, atol=0, rtol=0)
    assert out.spike_events is not None
    offsets = out.spike_events.offsets.detach().cpu()
    indices = out.spike_events.indices.detach().cpu()
    dense_spikes = expected_spikes.detach().cpu()
    for bucket in range(offsets.numel() - 1):
        start = int(offsets[bucket].item())
        end = int(offsets[bucket + 1].item())
        got = torch.sort(indices[start:end]).values
        expected = torch.nonzero(
            dense_spikes.reshape(-1, dense_spikes.shape[-1])[bucket],
            as_tuple=False,
        ).flatten()
        torch.testing.assert_close(got, expected.to(got.dtype))


def test_cuda_persistent_soft_reset_preserves_surplus_voltage():
    """Soft reset should subtract threshold delta rather than clamp to reset."""

    device = _require_cuda()
    x_seq = torch.tensor([[[2.0]]], device=device)
    events = _dense_to_events(x_seq)
    graph = EventCSRGraph(
        indptr=torch.tensor([0, 0], device=device, dtype=torch.int32),
        indices=torch.empty(0, device=device, dtype=torch.int32),
        weight=torch.empty(0, device=device, dtype=torch.float32),
        delay=None,
        shape=(1, 1),
    )
    state = PersistentSNNState(
        v=torch.zeros(1, 1, device=device),
        psc=torch.zeros(1, 1, device=device),
    )
    out = _run_cuda_or_skip(
        events,
        graph,
        state,
        PersistentSNNParams(window_size=1),
        return_mode="dense",
    )

    torch.testing.assert_close(out.spikes, torch.ones_like(out.spikes))
    torch.testing.assert_close(out.state.v, torch.ones_like(out.state.v))


def test_cuda_persistent_reusable_workspace_sums_duplicate_input_events():
    """A reused workspace should preserve duplicate-event accumulation.

    Two events target the same cell in one bucket. Running twice with the same
    scratch tensors verifies both the CAS accumulation and generation reset
    semantics without mutating the caller-owned initial state.
    """

    device = _require_cuda()
    graph = EventCSRGraph(
        indptr=torch.tensor([0, 0], device=device, dtype=torch.int32),
        indices=torch.empty(0, device=device, dtype=torch.int32),
        weight=torch.empty(0, device=device, dtype=torch.float32),
        delay=None,
        shape=(1, 1),
    )
    events = WindowedSpikeEvents(
        offsets=torch.tensor([0, 2], device=device, dtype=torch.int32),
        indices=torch.tensor([0, 0], device=device, dtype=torch.int32),
        values=torch.tensor([0.6, 0.7], device=device, dtype=torch.float32),
        shape=(1, 1, 1),
    )
    state = PersistentSNNState(
        v=torch.zeros(1, 1, device=device),
        psc=torch.zeros(1, 1, device=device),
    )
    workspace = make_persistent_snn_workspace(graph, batch_size=1)

    for _ in range(2):
        out = _run_cuda_or_skip(
            events,
            graph,
            state,
            PersistentSNNParams(window_size=1),
            return_mode="dense",
            workspace=workspace,
        )
        torch.testing.assert_close(out.spikes, torch.ones_like(out.spikes))
        torch.testing.assert_close(out.state.v, torch.full_like(out.state.v, 0.3))

    torch.testing.assert_close(state.v, torch.zeros_like(state.v))
    torch.testing.assert_close(state.psc, torch.zeros_like(state.psc))


@pytest.mark.parametrize(
    "fanouts",
    [
        (8, 17),
        (256, 257),
        (255, 256),
        (2050, 17),
    ],
)
def test_cuda_persistent_fanout_binning_matches_dense_reference(fanouts):
    """Opt-in fanout binning should preserve the baseline RSNN semantics.

    The cases cover all-low rows, all-high rows, the exact boundary where
    ``255`` stays low while ``256`` becomes high, and a long row spanning
    three 1024-edge tasks. Passing ``fanout_binning=True`` is the only switch
    needed by callers.
    """

    device = _require_cuda()
    graph, dense = _fanout_bucket_graph(device, fanouts)
    n_neuron = graph.shape[0]
    x_seq = torch.zeros(1, 1, n_neuron, device=device, dtype=torch.float32)
    x_seq[0, 0, : len(fanouts)] = 1.2
    state = PersistentSNNState(
        v=torch.zeros(1, n_neuron, device=device),
        psc=torch.zeros(1, n_neuron, device=device),
    )
    params = PersistentSNNParams(window_size=1)

    out = _run_cuda_or_skip(
        _dense_to_events(x_seq),
        graph,
        state,
        params,
        return_mode="both",
        fanout_binning=True,
    )
    ref_spikes, ref_v, ref_psc = _reference(x_seq, dense, state, params)

    assert out.spikes is not None
    assert out.spike_events is not None
    torch.testing.assert_close(out.spikes, ref_spikes, atol=0, rtol=0)
    torch.testing.assert_close(out.state.v, ref_v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.state.psc, ref_psc, atol=1e-5, rtol=1e-5)

    offsets = out.spike_events.offsets.detach().cpu()
    indices = torch.sort(out.spike_events.indices.detach().cpu()).values
    expected = torch.arange(len(fanouts), dtype=indices.dtype)
    assert offsets.tolist() == [0, len(fanouts)]
    torch.testing.assert_close(indices, expected)


@pytest.mark.parametrize(
    "fanouts",
    [
        (8, 17),
        (255, 256),
        (257, 2050),
    ],
)
def test_cuda_persistent_spike_block_matches_dense_reference(fanouts):
    """Spike-block tasks should preserve dynamics across the 256-edge split.

    Short rows from the same 32-neuron cell block share one queued task. The
    8-edge row uses direct relative-lane traversal while the 17-edge row uses
    a full warp. Rows at or above 256 edges become fixed 1024-edge segment
    tasks. These cases exercise both paths and the exact boundaries.
    """

    device = _require_cuda()
    graph, dense = _fanout_bucket_graph(device, fanouts)
    n_neuron = graph.shape[0]
    x_seq = torch.zeros(1, 1, n_neuron, device=device, dtype=torch.float32)
    x_seq[0, 0, : len(fanouts)] = 1.2
    state = PersistentSNNState(
        v=torch.zeros(1, n_neuron, device=device),
        psc=torch.zeros(1, n_neuron, device=device),
    )
    params = PersistentSNNParams(window_size=1)

    out = _run_cuda_or_skip(
        _dense_to_events(x_seq),
        graph,
        state,
        params,
        return_mode="both",
        spike_block=True,
    )
    ref_spikes, ref_v, ref_psc = _reference(x_seq, dense, state, params)

    assert out.spikes is not None
    assert out.spike_events is not None
    torch.testing.assert_close(out.spikes, ref_spikes, atol=0, rtol=0)
    torch.testing.assert_close(out.state.v, ref_v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.state.psc, ref_psc, atol=1e-5, rtol=1e-5)


def test_cuda_persistent_spike_block_merges_duplicate_posts():
    """Block and segment tasks should sum duplicate posts correctly.

    The first fired row is shorter than 256 edges and therefore travels in a
    BlockTask. The second has 300 edges and becomes one SegmentTask. Repeated
    destinations ensure atomic accumulation is exercised in both paths.
    """

    device = _require_cuda()
    n_neuron = 301
    low_count = 40
    high_count = 300
    graph = EventCSRGraph(
        indptr=torch.tensor(
            [0, low_count, low_count + high_count]
            + [low_count + high_count] * (n_neuron - 2),
            device=device,
            dtype=torch.int32,
        ),
        indices=torch.tensor(
            [7] * low_count + [9] * high_count,
            device=device,
            dtype=torch.int32,
        ),
        weight=torch.tensor(
            [0.01] * low_count + [0.002] * high_count,
            device=device,
            dtype=torch.float32,
        ),
        delay=None,
        shape=(n_neuron, n_neuron),
    )
    x_seq = torch.zeros(1, 1, n_neuron, device=device)
    x_seq[0, 0, :2] = 1.2
    state = PersistentSNNState(
        v=torch.zeros(1, n_neuron, device=device),
        psc=torch.zeros(1, n_neuron, device=device),
    )

    out = _run_cuda_or_skip(
        _dense_to_events(x_seq),
        graph,
        state,
        PersistentSNNParams(window_size=1),
        return_mode="dense",
        spike_block=True,
    )

    expected_psc = torch.zeros_like(state.psc)
    expected_psc[0, 7] = low_count * 0.01
    expected_psc[0, 9] = high_count * 0.002
    torch.testing.assert_close(out.state.psc, expected_psc, atol=1e-5, rtol=1e-5)


def test_cuda_persistent_spike_block_maps_nonzero_lane_rows():
    """A block command should retain the CSR row owned by a nonzero lane.

    Only lane 18 fires. This catches implementations that interpret the spike
    mask relative to the consuming warp rather than the queued neuron block
    and accidentally traverse row zero.
    """

    device = _require_cuda()
    n_neuron = 64
    posts = torch.arange(19, 35, device=device, dtype=torch.int32)
    graph = EventCSRGraph(
        indptr=torch.tensor(
            [0] * 19 + [posts.numel()] * (n_neuron - 18),
            device=device,
            dtype=torch.int32,
        ),
        indices=posts,
        weight=torch.ones(posts.numel(), device=device),
        delay=None,
        shape=(n_neuron, n_neuron),
    )
    x_seq = torch.zeros(1, 1, n_neuron, device=device)
    x_seq[0, 0, 18] = 1.2
    state = PersistentSNNState(
        v=torch.zeros(1, n_neuron, device=device),
        psc=torch.zeros(1, n_neuron, device=device),
    )

    out = _run_cuda_or_skip(
        _dense_to_events(x_seq),
        graph,
        state,
        PersistentSNNParams(window_size=1),
        return_mode="dense",
        spike_block=True,
    )

    expected_psc = torch.zeros_like(state.psc)
    expected_psc[0, posts.to(torch.long)] = 1.0
    torch.testing.assert_close(out.state.psc, expected_psc, atol=0, rtol=0)


@pytest.mark.parametrize("sort_row_edges", [False, True])
def test_cuda_persistent_physical_reorder_matches_original_ids(sort_row_edges):
    """A reordered multi-step window should restore the original API result.

    Random neuron IDs exercise the complete row/post permutation. Optional row
    sorting additionally changes physical edge order, while restoration must
    still recover dense spikes, final state, and per-bucket event identities.
    """

    device = _require_cuda()
    x_seq = torch.tensor(
        [
            [[1.2, 0.0, 1.3, 0.0]],
            [[0.0, 1.4, 0.0, 1.5]],
            [[0.8, 0.0, 0.9, 0.0]],
        ],
        device=device,
    )
    events = _dense_to_events(x_seq)
    graph, _dense = _graph(device)
    state = PersistentSNNState(
        v=torch.zeros(1, 4, device=device),
        psc=torch.zeros(1, 4, device=device),
    )
    params = PersistentSNNParams(window_size=3)
    reference = _run_cuda_or_skip(
        events,
        graph,
        state,
        params,
        return_mode="both",
        spike_block=True,
    )

    plan = build_persistent_snn_reorder_plan(
        graph,
        method="random",
        sort_row_edges=sort_row_edges,
        seed=5,
    )
    physical_output = _run_cuda_or_skip(
        reorder_windowed_spike_events(events, plan),
        plan.reordered_graph,
        reorder_persistent_snn_state(state, plan),
        params,
        return_mode="both",
        spike_block=True,
    )
    restored = restore_persistent_snn_output(physical_output, plan)

    assert reference.spikes is not None
    assert restored.spikes is not None
    torch.testing.assert_close(restored.spikes, reference.spikes, atol=0, rtol=0)
    torch.testing.assert_close(
        restored.state.v,
        reference.state.v,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        restored.state.psc,
        reference.state.psc,
        atol=1e-5,
        rtol=1e-5,
    )
    assert reference.spike_events is not None
    assert restored.spike_events is not None
    torch.testing.assert_close(
        restored.spike_events.offsets,
        reference.spike_events.offsets,
    )
    offsets = reference.spike_events.offsets
    for bucket in range(offsets.numel() - 1):
        start = int(offsets[bucket].item())
        end = int(offsets[bucket + 1].item())
        expected = torch.sort(reference.spike_events.indices[start:end]).values
        actual = torch.sort(restored.spike_events.indices[start:end]).values
        torch.testing.assert_close(actual, expected)


def test_cuda_persistent_rejects_multiple_task_schedulers():
    """Callers must select at most one experimental task scheduler."""

    device = _require_cuda()
    events = _dense_to_events(torch.ones(1, 1, 1, device=device))
    graph = EventCSRGraph(
        indptr=torch.tensor([0, 0], device=device, dtype=torch.int32),
        indices=torch.empty(0, device=device, dtype=torch.int32),
        weight=torch.empty(0, device=device, dtype=torch.float32),
        delay=None,
        shape=(1, 1),
    )
    state = PersistentSNNState(
        v=torch.zeros(1, 1, device=device),
        psc=torch.zeros(1, 1, device=device),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        persistent_snn_forward(
            events,
            graph,
            state,
            backend="cuda_persistent",
            fanout_binning=True,
            spike_block=True,
        )


def test_cuda_persistent_rejects_unsupported_v1_options():
    """Python dispatch should reject unsupported v1 options before JIT load."""

    device = _require_cuda()
    events = _dense_to_events(torch.ones(1, 1, 1, device=device))
    graph = EventCSRGraph(
        indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        indices=torch.tensor([0], device=device, dtype=torch.int32),
        weight=torch.tensor([0.0], device=device, dtype=torch.float32),
        delay=torch.tensor([1], device=device, dtype=torch.int32),
        shape=(1, 1),
    )
    state = PersistentSNNState(
        v=torch.zeros(1, 1, device=device),
        psc=torch.zeros(1, 1, device=device),
    )

    with pytest.raises(ValueError, match="hard_reset"):
        persistent_snn_forward(
            events,
            EventCSRGraph(
                indptr=graph.indptr,
                indices=graph.indices,
                weight=graph.weight,
                delay=None,
                shape=graph.shape,
            ),
            state,
            PersistentSNNParams(hard_reset=True),
            backend="cuda_persistent",
        )
    with pytest.raises(ValueError, match="refractory"):
        persistent_snn_forward(
            events,
            EventCSRGraph(
                indptr=graph.indptr,
                indices=graph.indices,
                weight=graph.weight,
                delay=None,
                shape=graph.shape,
            ),
            PersistentSNNState(
                v=state.v,
                psc=state.psc,
                refractory=torch.zeros_like(state.v),
            ),
            backend="cuda_persistent",
        )
    with pytest.raises(ValueError, match="nonzero delay"):
        persistent_snn_forward(
            events,
            graph,
            state,
            backend="cuda_persistent",
        )


def test_cuda_persistent_rejects_invalid_post_indices_after_mutation():
    """CSR validation should be cached safely and invalidated by mutation."""

    device = _require_cuda()
    events = _dense_to_events(torch.ones(1, 1, 1, device=device))
    indices = torch.tensor([0], device=device, dtype=torch.int32)
    graph = EventCSRGraph(
        indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        indices=indices,
        weight=torch.tensor([0.0], device=device, dtype=torch.float32),
        delay=None,
        shape=(1, 1),
    )
    state = PersistentSNNState(
        v=torch.zeros(1, 1, device=device),
        psc=torch.zeros(1, 1, device=device),
    )

    # The first call records that this tensor version is a valid CSR column
    # array. Mutating the same tensor increments Tensor._version, so the next
    # call must revalidate instead of trusting the cached result.
    persistent_snn_forward(
        events,
        graph,
        state,
        backend="cuda_persistent",
    )
    indices[0] = 1
    with pytest.raises(ValueError, match="graph.indices"):
        persistent_snn_forward(
            events,
            graph,
            state,
            backend="cuda_persistent",
        )
