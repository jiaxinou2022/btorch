"""Tests and executable examples for persistent physical graph reordering."""

import pytest
import torch

from btorch.backend.persistent.reorder import (
    ReorderConfig,
    build_neuron_permutation,
    dominant_post_blocks,
    prepare_reordered_inputs,
    reorder_graph,
    restore_output,
)
from btorch.backend.persistent_snn import (
    EventCSRGraph,
    PersistentSNNOutput,
    PersistentSNNState,
    WindowedSpikeEvents,
)


def _graph(device: torch.device | str = "cpu") -> EventCSRGraph:
    """Create a graph whose rows expose cost and similarity ordering."""

    rows = [
        [4, 5, 6],
        [0],
        [6, 7],
        [],
        [0, 1, 2],
        [6],
        [0, 1],
        [4],
    ]
    indptr = [0]
    indices = []
    for posts in rows:
        indices.extend(posts)
        indptr.append(len(indices))
    return EventCSRGraph(
        indptr=torch.tensor(indptr, device=device, dtype=torch.int32),
        indices=torch.tensor(indices, device=device, dtype=torch.int32),
        weight=torch.arange(
            1,
            len(indices) + 1,
            device=device,
            dtype=torch.float32,
        ),
        delay=torch.arange(len(indices), device=device, dtype=torch.int32),
        shape=(len(rows), len(rows)),
    )


@pytest.mark.parametrize(
    "mode",
    [
        "identity",
        "global_cost_similar",
        "global_cost_bucket",
        "global_cost_balanced",
        "local_cost_similar",
        "local_cost_bucket",
        "global_similarity",
        "global_cost_similarity",
        "global_similarity_cost",
        "local_similarity",
        "local_cost_similarity",
        "local_similarity_cost",
    ],
)
def test_build_permutation_is_bijective(mode):
    """Every experimental strategy must generate both inverse mappings."""

    graph = _graph()
    permutation = build_neuron_permutation(
        graph,
        ReorderConfig(
            mode=mode,
            local_window_size=4,
            extreme_fanout_threshold=3,
        ),
    )

    expected = torch.arange(graph.shape[0])
    torch.testing.assert_close(torch.sort(permutation.new_to_old).values, expected)
    torch.testing.assert_close(
        permutation.old_to_new[permutation.new_to_old],
        expected,
    )


def test_local_reorder_does_not_cross_windows():
    """Local sorting preserves window membership while changing row order."""

    graph = _graph()
    permutation = build_neuron_permutation(
        graph,
        ReorderConfig(
            mode="local_cost_similar",
            local_window_size=4,
            extreme_fanout_threshold=3,
        ),
    )

    old_window = permutation.new_to_old // 4
    new_window = torch.arange(graph.shape[0]) // 4
    torch.testing.assert_close(old_window, new_window)
    assert not torch.equal(permutation.new_to_old, torch.arange(graph.shape[0]))


def test_dominant_post_blocks_are_exact_and_deterministic():
    """Dominant blocks use exact counts, lowest-block ties, and -1 for empty."""

    dominant = dominant_post_blocks(_graph(), post_block_size=4)

    torch.testing.assert_close(
        dominant,
        torch.tensor([1, 0, 1, -1, 0, 1, 0, 1]),
    )


def test_reordered_csr_preserves_every_edge_and_edge_payload():
    """Physical row movement must remap both endpoints without losing edges."""

    graph = _graph()
    permutation = build_neuron_permutation(
        graph,
        ReorderConfig(
            mode="global_cost_similarity",
            extreme_fanout_threshold=3,
            post_block_size=4,
        ),
    )
    reordered = reorder_graph(graph, permutation)

    for new_pre, old_pre in enumerate(permutation.new_to_old.tolist()):
        old_begin = int(graph.indptr[old_pre])
        old_end = int(graph.indptr[old_pre + 1])
        new_begin = int(reordered.indptr[new_pre])
        new_end = int(reordered.indptr[new_pre + 1])
        expected_posts = permutation.old_to_new[
            graph.indices[old_begin:old_end].to(torch.int64)
        ]
        torch.testing.assert_close(
            reordered.indices[new_begin:new_end],
            expected_posts.to(torch.int32),
        )
        torch.testing.assert_close(
            reordered.weight[new_begin:new_end],
            graph.weight[old_begin:old_end],
        )
        torch.testing.assert_close(
            reordered.delay[new_begin:new_end],
            graph.delay[old_begin:old_end],
        )


def test_dynamic_inputs_and_outputs_round_trip():
    """Events, all state fields, dense spikes, and output events round-trip."""

    graph = _graph()
    events = WindowedSpikeEvents(
        offsets=torch.tensor([0, 3], dtype=torch.int32),
        indices=torch.tensor([0, 3, 7], dtype=torch.int32),
        values=torch.tensor([1.0, 2.0, 3.0]),
        shape=(1, 1, 8),
    )
    base = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    state = PersistentSNNState(
        v=base,
        psc=base + 10,
        refractory=base + 20,
        delay_ring=base.reshape(1, 1, 8) + 30,
    )
    prepared = prepare_reordered_inputs(
        events,
        graph,
        state,
        ReorderConfig(mode="global_cost_similarity", post_block_size=4),
    )
    output = PersistentSNNOutput(
        spikes=prepared.state.v.reshape(1, 1, 8),
        spike_events=prepared.events,
        state=prepared.state,
    )
    restored = restore_output(output, prepared.permutation)

    torch.testing.assert_close(restored.spikes, base.reshape(1, 1, 8))
    torch.testing.assert_close(restored.state.v, state.v)
    torch.testing.assert_close(restored.state.psc, state.psc)
    torch.testing.assert_close(restored.state.refractory, state.refractory)
    torch.testing.assert_close(restored.state.delay_ring, state.delay_ring)
    torch.testing.assert_close(restored.spike_events.indices, events.indices)
    assert restored.spike_events.offsets is events.offsets
    assert restored.spike_events.values is events.values


def test_reorder_rejects_non_square_graph():
    graph = EventCSRGraph(
        indptr=torch.tensor([0, 1, 1], dtype=torch.int32),
        indices=torch.tensor([2], dtype=torch.int32),
        weight=torch.ones(1),
        delay=None,
        shape=(2, 3),
    )

    with pytest.raises(ValueError, match="square"):
        build_neuron_permutation(graph)
