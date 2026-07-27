"""Physically reorder recurrent CSR graphs for persistent block kernels."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal

import torch

from btorch.backend.persistent_snn import (
    EventCSRGraph,
    PersistentSNNOutput,
    PersistentSNNState,
    WindowedSpikeEvents,
)


ReorderMode = Literal[
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
]


@dataclass(frozen=True)
class ReorderConfig:
    """Configure physical neuron and recurrent graph reordering.

    Args:
        mode: Ordering strategy. Local modes never move a neuron outside its
            original local window.
        neuron_block_size: Neurons grouped by one persistent BlockTask.
        local_window_size: Exchange window for local strategies.
        fanout_boundaries: Inclusive upper boundaries of coarse fanout buckets.
        extreme_fanout_threshold: Rows at or above this degree sort after
            ordinary rows within the selected global or local scope.
        post_block_size: Neurons in one post-synaptic similarity bucket.
    """

    mode: ReorderMode = "identity"
    neuron_block_size: int = 32
    local_window_size: int = 256
    fanout_boundaries: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 255)
    extreme_fanout_threshold: int = 256
    post_block_size: int = 32


@dataclass(frozen=True)
class NeuronPermutation:
    """Store both directions of a neuron permutation.

    ``new_to_old[new_id]`` identifies the original neuron at a reordered
    position. ``old_to_new[old_id]`` identifies its reordered position.
    """

    new_to_old: torch.Tensor
    old_to_new: torch.Tensor
    config: ReorderConfig


@dataclass(frozen=True)
class ReorderedPersistentInputs:
    """Hold a graph and dynamic inputs in the reordered numbering space."""

    events: WindowedSpikeEvents
    graph: EventCSRGraph
    state: PersistentSNNState
    permutation: NeuronPermutation


def _validate_config(config: ReorderConfig) -> None:
    if config.neuron_block_size <= 0:
        raise ValueError("neuron_block_size must be positive.")
    if config.local_window_size <= 0:
        raise ValueError("local_window_size must be positive.")
    if config.extreme_fanout_threshold <= 0:
        raise ValueError("extreme_fanout_threshold must be positive.")
    if config.post_block_size <= 0:
        raise ValueError("post_block_size must be positive.")
    if any(boundary < 0 for boundary in config.fanout_boundaries):
        raise ValueError("fanout_boundaries must be non-negative.")
    if tuple(sorted(set(config.fanout_boundaries))) != config.fanout_boundaries:
        raise ValueError("fanout_boundaries must be strictly increasing.")


def _validate_recurrent_graph(graph: EventCSRGraph) -> int:
    n_pre, n_post = graph.shape
    if n_pre != n_post:
        raise ValueError("Physical neuron reordering requires a square graph.")
    if graph.indptr.ndim != 1 or graph.indptr.numel() != n_pre + 1:
        raise ValueError("graph.indptr must have shape (N + 1,).")
    if graph.indices.ndim != 1 or graph.weight.shape != graph.indices.shape:
        raise ValueError("graph indices and weights must be matching vectors.")
    return n_pre


def dominant_post_blocks(
    graph: EventCSRGraph,
    *,
    post_block_size: int = 32,
) -> torch.Tensor:
    """Compute the exact dominant post-synaptic block for every CSR row.

    Empty rows use ``-1``. Ties select the lowest post block. The computation
    stays on the graph device and uses edge-sized temporary tensors.
    """

    n_neuron = _validate_recurrent_graph(graph)
    if post_block_size <= 0:
        raise ValueError("post_block_size must be positive.")
    device = graph.indptr.device
    edge_count = graph.indices.numel()
    dominant = torch.full((n_neuron,), -1, device=device, dtype=torch.int64)
    if edge_count == 0:
        return dominant

    degree = (graph.indptr[1:] - graph.indptr[:-1]).to(torch.int64)
    rows = torch.repeat_interleave(
        torch.arange(n_neuron, device=device, dtype=torch.int64),
        degree,
    )
    n_post_blocks = (n_neuron + post_block_size - 1) // post_block_size
    post_blocks = graph.indices.to(torch.int64) // post_block_size
    encoded = rows * n_post_blocks + post_blocks
    pairs, counts = torch.unique(encoded, sorted=True, return_counts=True)
    pair_rows = pairs // n_post_blocks
    pair_blocks = pairs % n_post_blocks

    max_counts = torch.zeros(n_neuron, device=device, dtype=counts.dtype)
    max_counts.scatter_reduce_(
        0,
        pair_rows,
        counts,
        reduce="amax",
        include_self=False,
    )
    candidates = torch.where(
        counts == max_counts[pair_rows],
        pair_blocks,
        torch.full_like(pair_blocks, n_post_blocks),
    )
    dominant.fill_(n_post_blocks)
    dominant.scatter_reduce_(
        0,
        pair_rows,
        candidates,
        reduce="amin",
        include_self=True,
    )
    return torch.where(dominant == n_post_blocks, -1, dominant)


def build_neuron_permutation(
    graph: EventCSRGraph,
    config: ReorderConfig | None = None,
) -> NeuronPermutation:
    """Build a stable CPU-planned permutation and copy it to the graph device."""

    config = config or ReorderConfig()
    _validate_config(config)
    n_neuron = _validate_recurrent_graph(graph)
    degree = (graph.indptr[1:] - graph.indptr[:-1]).to("cpu", torch.int64).tolist()
    needs_similarity = "similarity" in config.mode
    dominant = (
        dominant_post_blocks(
            graph,
            post_block_size=config.post_block_size,
        )
        .cpu()
        .tolist()
        if needs_similarity
        else [-1] * n_neuron
    )

    def bucket(old_id: int) -> int:
        return bisect_left(config.fanout_boundaries, degree[old_id])

    def extreme(old_id: int) -> int:
        return int(degree[old_id] >= config.extreme_fanout_threshold)

    def strategy_key(old_id: int) -> tuple[int, ...]:
        if config.mode.endswith("cost_similarity"):
            return extreme(old_id), bucket(old_id), dominant[old_id], old_id
        if config.mode.endswith("similarity_cost"):
            return extreme(old_id), dominant[old_id], bucket(old_id), old_id
        if config.mode.endswith("cost_similar"):
            return extreme(old_id), degree[old_id], old_id
        if config.mode.endswith("cost_bucket"):
            return extreme(old_id), bucket(old_id), old_id
        if config.mode.endswith("similarity"):
            return extreme(old_id), dominant[old_id], old_id
        return (old_id,)

    if config.mode == "identity":
        order = list(range(n_neuron))
    elif config.mode == "global_cost_balanced":
        ordinary = [
            old_id
            for old_id in range(n_neuron)
            if not extreme(old_id)
        ]
        extremes = [
            old_id
            for old_id in range(n_neuron)
            if extreme(old_id)
        ]
        ordinary.sort(key=lambda old_id: (degree[old_id], old_id))
        extremes.sort(key=lambda old_id: (degree[old_id], old_id))
        balanced: list[int] = []
        left, right = 0, len(ordinary) - 1
        while left <= right:
            balanced.append(ordinary[right])
            right -= 1
            if left <= right:
                balanced.append(ordinary[left])
                left += 1
        order = balanced + extremes
    elif config.mode.startswith("local_"):
        order = []
        for start in range(0, n_neuron, config.local_window_size):
            window = list(
                range(start, min(start + config.local_window_size, n_neuron))
            )
            order.extend(sorted(window, key=strategy_key))
    else:
        order = sorted(range(n_neuron), key=strategy_key)

    new_to_old = torch.tensor(order, device=graph.indptr.device, dtype=torch.int64)
    old_to_new = torch.empty_like(new_to_old)
    old_to_new[new_to_old] = torch.arange(
        n_neuron,
        device=graph.indptr.device,
        dtype=torch.int64,
    )
    return NeuronPermutation(
        new_to_old=new_to_old,
        old_to_new=old_to_new,
        config=config,
    )


def reorder_graph(
    graph: EventCSRGraph,
    permutation: NeuronPermutation,
) -> EventCSRGraph:
    """Physically move CSR rows and remap their post-synaptic indices."""

    n_neuron = _validate_recurrent_graph(graph)
    if permutation.new_to_old.numel() != n_neuron:
        raise ValueError("Permutation size must match graph shape.")
    new_to_old = permutation.new_to_old
    old_to_new = permutation.old_to_new
    old_degree = (graph.indptr[1:] - graph.indptr[:-1]).to(torch.int64)
    new_degree = old_degree[new_to_old]
    new_indptr64 = torch.zeros(
        n_neuron + 1,
        device=graph.indptr.device,
        dtype=torch.int64,
    )
    new_indptr64[1:] = torch.cumsum(new_degree, dim=0)

    if graph.indices.numel():
        new_rows = torch.repeat_interleave(
            torch.arange(n_neuron, device=graph.indptr.device),
            new_degree,
        )
        local_edge = torch.arange(
            graph.indices.numel(),
            device=graph.indptr.device,
            dtype=torch.int64,
        ) - new_indptr64[new_rows]
        source_edge = graph.indptr[new_to_old[new_rows]].to(torch.int64) + local_edge
        new_indices = old_to_new[graph.indices[source_edge].to(torch.int64)]
        new_weight = graph.weight[source_edge]
        new_delay = graph.delay[source_edge] if graph.delay is not None else None
    else:
        new_indices = graph.indices.clone()
        new_weight = graph.weight.clone()
        new_delay = graph.delay.clone() if graph.delay is not None else None

    return EventCSRGraph(
        indptr=new_indptr64.to(graph.indptr.dtype).contiguous(),
        indices=new_indices.to(graph.indices.dtype).contiguous(),
        weight=new_weight.contiguous(),
        delay=new_delay.contiguous() if new_delay is not None else None,
        shape=graph.shape,
    )


def reorder_events(
    events: WindowedSpikeEvents,
    permutation: NeuronPermutation,
) -> WindowedSpikeEvents:
    """Remap event neuron indices while preserving bucket and event order."""

    if events.shape[2] != permutation.old_to_new.numel():
        raise ValueError("Event neuron count must match permutation size.")
    indices = permutation.old_to_new[events.indices.to(torch.int64)]
    return WindowedSpikeEvents(
        offsets=events.offsets,
        indices=indices.to(events.indices.dtype).contiguous(),
        values=events.values,
        shape=events.shape,
    )


def reorder_state(
    state: PersistentSNNState,
    permutation: NeuronPermutation,
) -> PersistentSNNState:
    """Move neuron-indexed state into the reordered numbering space."""

    index = permutation.new_to_old

    def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return tensor.index_select(-1, index) if tensor is not None else None

    return PersistentSNNState(
        v=move(state.v),
        psc=move(state.psc),
        refractory=move(state.refractory),
        delay_ring=move(state.delay_ring),
    )


def restore_state(
    state: PersistentSNNState,
    permutation: NeuronPermutation,
) -> PersistentSNNState:
    """Restore neuron-indexed state to the original numbering space."""

    index = permutation.old_to_new

    def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return tensor.index_select(-1, index) if tensor is not None else None

    return PersistentSNNState(
        v=move(state.v),
        psc=move(state.psc),
        refractory=move(state.refractory),
        delay_ring=move(state.delay_ring),
    )


def restore_output(
    output: PersistentSNNOutput,
    permutation: NeuronPermutation,
) -> PersistentSNNOutput:
    """Restore dense spikes, event indices, and final state numbering."""

    spikes = (
        output.spikes.index_select(-1, permutation.old_to_new)
        if output.spikes is not None
        else None
    )
    spike_events = output.spike_events
    if spike_events is not None:
        restored_indices = permutation.new_to_old[
            spike_events.indices.to(torch.int64)
        ]
        spike_events = WindowedSpikeEvents(
            offsets=spike_events.offsets,
            indices=restored_indices.to(spike_events.indices.dtype).contiguous(),
            values=spike_events.values,
            shape=spike_events.shape,
        )
    return PersistentSNNOutput(
        spikes=spikes,
        spike_events=spike_events,
        state=restore_state(output.state, permutation),
    )


def prepare_reordered_inputs(
    events: WindowedSpikeEvents,
    graph: EventCSRGraph,
    state: PersistentSNNState,
    config: ReorderConfig | None = None,
) -> ReorderedPersistentInputs:
    """Build one permutation and transform all persistent-kernel inputs."""

    permutation = build_neuron_permutation(graph, config)
    return ReorderedPersistentInputs(
        events=reorder_events(events, permutation),
        graph=reorder_graph(graph, permutation),
        state=reorder_state(state, permutation),
        permutation=permutation,
    )


__all__ = [
    "NeuronPermutation",
    "ReorderConfig",
    "ReorderMode",
    "ReorderedPersistentInputs",
    "build_neuron_permutation",
    "dominant_post_blocks",
    "prepare_reordered_inputs",
    "reorder_events",
    "reorder_graph",
    "reorder_state",
    "restore_output",
    "restore_state",
]
