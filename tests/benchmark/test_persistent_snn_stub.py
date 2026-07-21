import pytest
import torch

from btorch.backend.persistent_snn import (
    EventCSRGraph,
    PersistentSNNParams,
    PersistentSNNState,
    WindowedSpikeEvents,
    build_persistent_snn_reorder_plan,
    make_empty_state,
    persistent_snn_forward,
    reorder_persistent_snn_state,
    reorder_windowed_spike_events,
    restore_persistent_snn_state,
)


def _make_events(
    *,
    t_steps: int,
    batch_size: int,
    n_pre: int,
    events_per_bucket: int,
    device: str = "cpu",
) -> WindowedSpikeEvents:
    """Create a deterministic windowed event tensor for contract tests."""

    n_buckets = t_steps * batch_size
    offsets = torch.arange(
        n_buckets + 1,
        device=device,
        dtype=torch.int32,
    )
    offsets = offsets * events_per_bucket
    nnz = int(offsets[-1].item())
    indices = torch.arange(nnz, device=device, dtype=torch.int32) % n_pre
    values = torch.ones(nnz, device=device, dtype=torch.float32)
    return WindowedSpikeEvents(
        offsets=offsets,
        indices=indices,
        values=values,
        shape=(t_steps, batch_size, n_pre),
    )


def _make_graph(
    *,
    n_pre: int,
    n_post: int,
    fanout: int,
    device: str = "cpu",
) -> EventCSRGraph:
    """Create a deterministic pre-row CSR graph for contract tests."""

    indptr = torch.arange(n_pre + 1, device=device, dtype=torch.int32) * fanout
    nnz = int(indptr[-1].item())
    indices = torch.arange(nnz, device=device, dtype=torch.int32) % n_post
    weight = torch.linspace(0.1, 1.0, nnz, device=device, dtype=torch.float32)
    delay = torch.zeros(nnz, device=device, dtype=torch.int32)
    return EventCSRGraph(
        indptr=indptr,
        indices=indices,
        weight=weight,
        delay=delay,
        shape=(n_pre, n_post),
    )


def test_persistent_snn_stub_returns_zero_dense_spikes_and_preserves_state():
    """The empty operator must preserve state and emit no dense spikes.

    This is the correctness proof for the scaffold: before a CUDA kernel is
    implemented, the mathematical operation is the identity on state and the
    zero map on emitted spikes.
    """

    t_steps, batch_size, n_pre, n_post = 5, 2, 7, 11
    events = _make_events(
        t_steps=t_steps,
        batch_size=batch_size,
        n_pre=n_pre,
        events_per_bucket=3,
    )
    graph = _make_graph(n_pre=n_pre, n_post=n_post, fanout=4)
    state = make_empty_state(batch_size, n_post)
    state.v.add_(0.25)
    state.psc.add_(0.5)
    assert state.refractory is not None
    state.refractory.add_(1.0)

    out = persistent_snn_forward(
        events,
        graph,
        state,
        PersistentSNNParams(window_size=t_steps),
        backend="torch_stub",
        return_mode="dense",
    )

    assert out.spikes is not None
    assert out.spike_events is None
    assert out.spikes.shape == (t_steps, batch_size, n_post)
    assert torch.count_nonzero(out.spikes) == 0
    torch.testing.assert_close(out.state.v, state.v)
    torch.testing.assert_close(out.state.psc, state.psc)
    torch.testing.assert_close(out.state.refractory, state.refractory)


def test_persistent_snn_stub_returns_empty_event_output():
    """Event-mode output must be a valid empty WindowedSpikeEvents object."""

    events = _make_events(t_steps=4, batch_size=3, n_pre=5, events_per_bucket=2)
    graph = _make_graph(n_pre=5, n_post=6, fanout=3)
    state = make_empty_state(3, 6)

    out = persistent_snn_forward(
        events,
        graph,
        state,
        backend="torch_stub",
        return_mode="both",
    )

    assert out.spikes is not None
    assert out.spike_events is not None
    assert out.spike_events.shape == (4, 3, 6)
    assert out.spike_events.indices.numel() == 0
    assert torch.count_nonzero(out.spike_events.offsets) == 0


def test_persistent_snn_stub_validates_shape_mismatch():
    """Validation should fail early when graph and event neuron counts differ."""

    events = _make_events(t_steps=2, batch_size=1, n_pre=5, events_per_bucket=1)
    graph = _make_graph(n_pre=4, n_post=6, fanout=2)
    state = make_empty_state(1, 6)

    with pytest.raises(ValueError, match="N_pre"):
        persistent_snn_forward(events, graph, state, backend="torch_stub")


@pytest.mark.parametrize(
    "method",
    [
        "identity",
        "random",
        "global_fanout",
        "local_fanout",
        "global_primary_tile",
        "local_primary_tile",
    ],
)
def test_persistent_reorder_plan_rebuilds_equivalent_csr(method):
    """Every physical CSR edge should map back to its original directed edge.

    The graph uses unequal row lengths and nontrivial post IDs so this test
    verifies row permutation, post-index rewriting, weight movement, delay
    movement, and the inverse permutation rather than only checking shapes.
    """

    indptr = torch.tensor([0, 2, 3, 6, 6, 8], dtype=torch.int32)
    indices = torch.tensor([4, 1, 3, 0, 4, 2, 1, 3], dtype=torch.int32)
    weights = torch.arange(1, 9, dtype=torch.float32)
    delays = torch.arange(8, dtype=torch.int32)
    graph = EventCSRGraph(
        indptr=indptr,
        indices=indices,
        weight=weights,
        delay=delays,
        shape=(5, 5),
    )

    plan = build_persistent_snn_reorder_plan(
        graph,
        method=method,
        region_size=3,
        post_tile_size=2,
        sort_row_edges=False,
        seed=7,
    )

    expected_ids = torch.arange(5)
    torch.testing.assert_close(
        plan.old_to_new.index_select(0, plan.new_to_old),
        expected_ids,
    )
    assert plan.reordered_graph.indices.numel() == graph.indices.numel()
    assert plan.reordered_graph.delay is not None

    # A physical row corresponds to exactly one original row. Its post IDs are
    # converted back here before comparing the edge payload in original order.
    for new_pre, old_pre_tensor in enumerate(plan.new_to_old):
        old_pre = int(old_pre_tensor.item())
        old_start = int(graph.indptr[old_pre].item())
        old_end = int(graph.indptr[old_pre + 1].item())
        new_start = int(plan.reordered_graph.indptr[new_pre].item())
        new_end = int(plan.reordered_graph.indptr[new_pre + 1].item())
        restored_posts = plan.new_to_old.index_select(
            0,
            plan.reordered_graph.indices[new_start:new_end].to(torch.long),
        )
        torch.testing.assert_close(
            restored_posts,
            graph.indices[old_start:old_end].to(torch.long),
        )
        torch.testing.assert_close(
            plan.reordered_graph.weight[new_start:new_end],
            graph.weight[old_start:old_end],
        )
        torch.testing.assert_close(
            plan.reordered_graph.delay[new_start:new_end],
            graph.delay[old_start:old_end],
        )


def test_persistent_reorder_state_and_events_round_trip():
    """State and sparse event IDs should survive a physical-ID round trip."""

    graph = _make_graph(n_pre=7, n_post=7, fanout=3)
    plan = build_persistent_snn_reorder_plan(graph, method="random", seed=11)
    state = PersistentSNNState(
        v=torch.arange(14, dtype=torch.float32).reshape(2, 7),
        psc=torch.arange(14, 28, dtype=torch.float32).reshape(2, 7),
        refractory=torch.arange(28, 42, dtype=torch.float32).reshape(2, 7),
    )
    events = _make_events(
        t_steps=2,
        batch_size=2,
        n_pre=7,
        events_per_bucket=3,
    )

    restored_state = restore_persistent_snn_state(
        reorder_persistent_snn_state(state, plan),
        plan,
    )
    torch.testing.assert_close(restored_state.v, state.v)
    torch.testing.assert_close(restored_state.psc, state.psc)
    torch.testing.assert_close(restored_state.refractory, state.refractory)

    reordered_events = reorder_windowed_spike_events(events, plan)
    restored_indices = plan.new_to_old.index_select(
        0,
        reordered_events.indices.to(torch.long),
    )
    torch.testing.assert_close(restored_indices, events.indices.to(torch.long))
    assert reordered_events.offsets.data_ptr() == events.offsets.data_ptr()
    assert reordered_events.values is events.values


def test_persistent_reorder_can_sort_each_physical_csr_row():
    """Optional edge sorting should make every physical post row monotonic."""

    graph = _make_graph(n_pre=9, n_post=9, fanout=5)
    plan = build_persistent_snn_reorder_plan(
        graph,
        method="random",
        sort_row_edges=True,
        seed=3,
    )

    for row in range(graph.shape[0]):
        start = int(plan.reordered_graph.indptr[row].item())
        end = int(plan.reordered_graph.indptr[row + 1].item())
        posts = plan.reordered_graph.indices[start:end]
        assert torch.all(posts[1:] >= posts[:-1])


def test_persistent_reorder_defaults_to_measured_layout():
    """Default preprocessing should use the selected stable benchmark winner."""

    graph = _make_graph(n_pre=129, n_post=129, fanout=2)
    plan = build_persistent_snn_reorder_plan(graph)

    assert plan.method == "local_fanout"
    assert plan.region_size == 128
    assert plan.post_tile_size == 128
    assert plan.sort_row_edges
