"""Persistent SNN operator scaffold.

This module defines the stable Python-side contract for an event-driven
persistent-kernel SNN backend. The current implementation is intentionally a
no-op reference/stub: it validates layout, preserves state, and returns empty
spike outputs. A future C++/CUDA implementation can register
``torch.ops.btorch_cuda.persistent_snn_forward`` without changing benchmark
or caller code.

The first persistent kernel target is intentionally narrow: recurrent RSNN
dynamics with a scalar-parameter LIF neuron and an ``ExponentialPSC`` synapse.
The benchmark and persistent contract share these defaults:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - First-kernel meaning
   * - ``dt``
     - ``1.0``
     - Euler step size for LIF and exponential PSC decay.
   * - ``tau_mem``
     - ``20.0``
     - LIF membrane time constant.
   * - ``tau_syn``
     - ``5.0``
     - Exponential PSC time constant.
   * - ``v_threshold``
     - ``1.0``
     - Spike threshold; spikes are emitted when ``v >= v_threshold``.
   * - ``v_reset``
     - ``0.0``
     - Reset baseline used by the LIF leak and reset delta.
   * - ``c_m``
     - ``1.0``
     - Membrane capacitance divisor for input current.
   * - ``hard_reset``
     - ``False``
     - Soft reset: subtract ``v_threshold - v_reset`` after a spike.
   * - ``window_size``
     - ``128``
     - Suggested processing window length in time steps.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Literal

import torch


Backend = Literal["auto", "torch_stub", "cuda_persistent"]
ReturnMode = Literal["dense", "events", "both"]
ReorderMethod = Literal[
    "identity",
    "random",
    "global_fanout",
    "local_fanout",
    "global_primary_tile",
    "local_primary_tile",
]
_FANOUT_BINNING_THRESHOLD = 256


# Cache the two scalar boundary checks for immutable/reused event layouts.
# Tensor._version changes after in-place mutation, so cached validation cannot
# silently accept modified offsets.
_event_offsets_cache: dict[
    int, tuple["weakref.ref[torch.Tensor]", int, int]
] = {}


# Post-synaptic indices are fixed CSR structure in normal use. Validate their
# range once, then invalidate the result after any in-place tensor mutation.
_graph_indices_cache: dict[
    int, tuple["weakref.ref[torch.Tensor]", int, int]
] = {}


def _event_offsets_validated(offsets: torch.Tensor, nnz: int) -> bool:
    cached = _event_offsets_cache.get(id(offsets))
    return (
        cached is not None
        and cached[0]() is offsets
        and cached[1] == nnz
        and cached[2] == offsets._version
    )


def _remember_event_offsets(offsets: torch.Tensor, nnz: int) -> None:
    key = id(offsets)

    def _evict(_: object, key: int = key) -> None:
        _event_offsets_cache.pop(key, None)

    _event_offsets_cache[key] = (weakref.ref(offsets, _evict), nnz, offsets._version)


def _graph_indices_validated(indices: torch.Tensor, n_post: int) -> bool:
    cached = _graph_indices_cache.get(id(indices))
    return (
        cached is not None
        and cached[0]() is indices
        and cached[1] == n_post
        and cached[2] == indices._version
    )


def _remember_graph_indices(indices: torch.Tensor, n_post: int) -> None:
    key = id(indices)

    def _evict(_: object, key: int = key) -> None:
        _graph_indices_cache.pop(key, None)

    _graph_indices_cache[key] = (
        weakref.ref(indices, _evict),
        n_post,
        indices._version,
    )


def _validate_graph_indices(indices: torch.Tensor, n_post: int) -> None:
    if _graph_indices_validated(indices, n_post):
        return
    valid = torch.logical_and(indices >= 0, indices < n_post).all().item()
    if not valid:
        raise ValueError("graph.indices must be in the range [0, N_post).")
    _remember_graph_indices(indices, n_post)


@dataclass(frozen=True)
class WindowedSpikeEvents:
    """Time-batch bucketed spike-event input.

    Args:
        offsets: Prefix sum over ``T * B`` buckets, shape ``(T * B + 1,)``.
        indices: Pre-synaptic neuron indices, shape ``(nnz,)``.
        values: Optional per-event values, shape ``(nnz,)``.
        shape: Logical dense event shape ``(T, B, N_pre)``.
    """

    offsets: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor | None
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class EventCSRGraph:
    """Pre-synaptic-row CSR graph for event fanout.

    Args:
        indptr: Row pointer for pre-synaptic neurons, shape ``(N_pre + 1,)``.
        indices: Post-synaptic neuron indices, shape ``(E,)``.
        weight: Edge weights, shape ``(E,)``.
        delay: Optional delay in integer time steps, shape ``(E,)``.
        shape: Logical graph shape ``(N_pre, N_post)``.
    """

    indptr: torch.Tensor
    indices: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor | None
    shape: tuple[int, int]


@dataclass(frozen=True)
class PersistentSNNState:
    """State tensors carried across persistent SNN windows."""

    v: torch.Tensor
    psc: torch.Tensor
    refractory: torch.Tensor | None = None
    delay_ring: torch.Tensor | None = None


@dataclass(frozen=True)
class PersistentSNNWorkspace:
    """Hold reusable CUDA scratch tensors for steady-state window execution.

    A workspace is mutable scratch storage. Do not reuse the same instance in
    overlapping forwards or on multiple CUDA streams concurrently.
    """

    input_current: torch.Tensor
    queue_batch: torch.Tensor
    queue_edge_start: torch.Tensor
    queue_edge_end: torch.Tensor
    spike_count: torch.Tensor
    work_counter: torch.Tensor


@dataclass(frozen=True)
class PersistentSNNOutput:
    """Output of the persistent SNN operator scaffold."""

    spikes: torch.Tensor | None
    spike_events: WindowedSpikeEvents | None
    state: PersistentSNNState


@dataclass(frozen=True)
class PersistentSNNReorderPlan:
    """Hold one reusable physical neuron and CSR permutation.

    Args:
        new_to_old: Original neuron ID for each physical neuron ID.
        old_to_new: Physical neuron ID for each original neuron ID.
        reordered_graph: CSR graph stored entirely in physical ID space.
        method: Strategy used to construct the permutation.
        region_size: Maximum old-ID region sorted independently.
        post_tile_size: Number of post neurons in one locality tile.
        sort_row_edges: Whether each rebuilt row is sorted by physical post ID.
    """

    new_to_old: torch.Tensor
    old_to_new: torch.Tensor
    reordered_graph: EventCSRGraph
    method: ReorderMethod
    region_size: int
    post_tile_size: int
    sort_row_edges: bool


@dataclass(frozen=True)
class PersistentSNNParams:
    """Scalar simulation parameters for the persistent SNN operator.

    The first CUDA persistent kernel is scoped to scalar-parameter
    LIF + ExponentialPSC dynamics. Its default reset mode is the same as
    :class:`btorch.models.neurons.lif.LIF`: soft reset
    (``hard_reset=False``).
    """

    dt: float = 1.0
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    c_m: float = 1.0
    hard_reset: bool = False
    window_size: int = 128


def _check_int_tensor(name: str, tensor: torch.Tensor, ndim: int) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.ndim}.")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64, got {tensor.dtype}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_events(events: WindowedSpikeEvents) -> tuple[int, int, int]:
    offsets, indices = events.offsets, events.indices
    t_steps, batch_size, n_pre = events.shape
    if t_steps <= 0 or batch_size <= 0 or n_pre <= 0:
        raise ValueError(f"events.shape must be positive, got {events.shape}.")
    _check_int_tensor("events.offsets", offsets, ndim=1)
    _check_int_tensor("events.indices", indices, ndim=1)
    if offsets.numel() != t_steps * batch_size + 1:
        raise ValueError(
            "events.offsets must have shape (T * B + 1,), got "
            f"{tuple(offsets.shape)} for shape={events.shape}."
        )
    if not _event_offsets_validated(offsets, indices.numel()):
        # Pay one D2H sync only when a new or in-place-modified layout appears.
        offsets_first, offsets_last = offsets[[0, -1]].tolist()
        if offsets_first != 0:
            raise ValueError("events.offsets[0] must be zero.")
        if offsets_last != indices.numel():
            raise ValueError("events.offsets[-1] must equal events.indices.numel().")
        _remember_event_offsets(offsets, indices.numel())
    if events.values is not None:
        if events.values.shape != indices.shape:
            raise ValueError("events.values must have the same shape as indices.")
        if events.values.device != indices.device:
            raise ValueError("events.values and indices must be on the same device.")
    return t_steps, batch_size, n_pre


def _validate_graph(graph: EventCSRGraph, n_pre: int) -> int:
    graph_n_pre, n_post = graph.shape
    if graph_n_pre != n_pre:
        raise ValueError(f"graph has N_pre={graph_n_pre}, events have N_pre={n_pre}.")
    if n_post <= 0:
        raise ValueError(f"graph N_post must be positive, got {n_post}.")
    _check_int_tensor("graph.indptr", graph.indptr, ndim=1)
    _check_int_tensor("graph.indices", graph.indices, ndim=1)
    if graph.indptr.numel() != n_pre + 1:
        raise ValueError("graph.indptr must have shape (N_pre + 1,).")
    if graph.weight.ndim != 1:
        raise ValueError("graph.weight must have shape (E,).")
    if graph.indices.shape != graph.weight.shape:
        raise ValueError("graph.indices and graph.weight must have matching shape.")
    if graph.delay is not None and graph.delay.shape != graph.indices.shape:
        raise ValueError("graph.delay must have the same shape as graph.indices.")
    return n_post


def _validate_state(
    state: PersistentSNNState,
    *,
    batch_size: int,
    n_post: int,
) -> None:
    expected = (batch_size, n_post)
    if tuple(state.v.shape) != expected:
        raise ValueError(f"state.v must have shape {expected}, got {state.v.shape}.")
    if tuple(state.psc.shape) != expected:
        raise ValueError(
            f"state.psc must have shape {expected}, got {state.psc.shape}."
        )
    if state.refractory is not None and tuple(state.refractory.shape) != expected:
        raise ValueError(
            f"state.refractory must have shape {expected}, "
            f"got {state.refractory.shape}."
        )


def make_empty_state(
    batch_size: int,
    n_neuron: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    refractory: bool = True,
    delay_ring_shape: tuple[int, int, int] | None = None,
) -> PersistentSNNState:
    """Create zero-filled persistent SNN state tensors."""

    v = torch.zeros((batch_size, n_neuron), device=device, dtype=dtype)
    psc = torch.zeros_like(v)
    refractory_t = torch.zeros_like(v) if refractory else None
    delay_ring = None
    if delay_ring_shape is not None:
        delay_ring = torch.zeros(delay_ring_shape, device=device, dtype=dtype)
    return PersistentSNNState(
        v=v,
        psc=psc,
        refractory=refractory_t,
        delay_ring=delay_ring,
    )


def make_persistent_snn_workspace(
    graph: EventCSRGraph,
    batch_size: int,
) -> PersistentSNNWorkspace:
    """Create reusable scratch storage for the CUDA persistent backend"""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    n_neuron = graph.shape[0]
    edge_count = graph.indices.numel()
    # This single workspace serves every kernel variant. The spike-block
    # variant needs one task per 32-cell block plus up to one rounding task per
    # CSR row when 1024-edge segments are used.
    queue_capacity = batch_size * (
        (n_neuron + 31) // 32 + n_neuron + (edge_count + 1023) // 1024
    )
    int_options = {"device": graph.indptr.device, "dtype": torch.int32}
    return PersistentSNNWorkspace(
        input_current=torch.zeros(
            batch_size,
            n_neuron,
            device=graph.indptr.device,
            dtype=torch.float32,
        ),
        queue_batch=torch.empty(queue_capacity, **int_options),
        queue_edge_start=torch.empty(queue_capacity, **int_options),
        queue_edge_end=torch.empty(queue_capacity, **int_options),
        # The binned kernel uses index 0 for long-row segments and index 1 for
        # ordinary rows. The non-binned kernel only touches index 0.
        spike_count=torch.empty(2, **int_options),
        work_counter=torch.empty(2, **int_options),
    )


def _primary_post_tiles(graph: EventCSRGraph, post_tile_size: int) -> torch.Tensor:
    """Return the most populated post tile for every CSR row."""

    n_neuron = graph.shape[0]
    n_tiles = (graph.shape[1] + post_tile_size - 1) // post_tile_size
    degrees = (graph.indptr[1:] - graph.indptr[:-1]).to(torch.long)
    edge_rows = torch.repeat_interleave(
        torch.arange(n_neuron, device=graph.indptr.device),
        degrees,
    )
    post_tiles = torch.div(
        graph.indices.to(torch.long),
        post_tile_size,
        rounding_mode="floor",
    )
    histogram = torch.bincount(
        edge_rows * n_tiles + post_tiles,
        minlength=n_neuron * n_tiles,
    ).reshape(n_neuron, n_tiles)
    return histogram.argmax(dim=1)


def _build_new_to_old(
    graph: EventCSRGraph,
    method: ReorderMethod,
    region_size: int,
    post_tile_size: int,
    seed: int,
) -> torch.Tensor:
    """Construct a physical-to-original neuron permutation."""

    n_neuron = graph.shape[0]
    device = graph.indptr.device
    if method == "identity":
        return torch.arange(n_neuron, device=device, dtype=torch.long)
    if method == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randperm(n_neuron, generator=generator).to(device)

    degrees = (graph.indptr[1:] - graph.indptr[:-1]).detach().cpu().tolist()
    primary_tiles: list[int] | None = None
    if method in ("global_primary_tile", "local_primary_tile"):
        primary_tiles = (
            _primary_post_tiles(graph, post_tile_size).detach().cpu().tolist()
        )

    local = method in ("local_fanout", "local_primary_tile")
    span = region_size if local else n_neuron
    permutation: list[int] = []
    for region_start in range(0, n_neuron, span):
        region_end = min(region_start + span, n_neuron)
        neurons = range(region_start, region_end)
        if primary_tiles is None:
            ordered = sorted(neurons, key=lambda neuron: (degrees[neuron], neuron))
        else:
            ordered = sorted(
                neurons,
                key=lambda neuron: (
                    primary_tiles[neuron],
                    degrees[neuron],
                    neuron,
                ),
            )
        permutation.extend(ordered)
    return torch.tensor(permutation, device=device, dtype=torch.long)


def build_persistent_snn_reorder_plan(
    graph: EventCSRGraph,
    method: ReorderMethod = "local_fanout",
    *,
    region_size: int = 128,
    post_tile_size: int = 128,
    sort_row_edges: bool = True,
    seed: int = 0,
) -> PersistentSNNReorderPlan:
    """Build a reusable physical neuron and CSR reorder plan.

    The returned graph uses physical IDs for both CSR rows and post indices.
    Building the plan is an explicit preprocessing operation and is never
    performed by :func:`persistent_snn_forward`.

    Args:
        graph: Square recurrent graph in the caller's original neuron IDs.
        method: Permutation strategy.
        region_size: Old-ID region sorted independently by local strategies.
        post_tile_size: Post-ID tile width used by post-aware strategies.
        sort_row_edges: Sort rebuilt rows by physical post ID.
        seed: CPU random seed used only by the random strategy.

    Returns:
        Reusable mappings and rebuilt physical-ID CSR graph.

    Raises:
        ValueError: If the graph is not square or a parameter is invalid.
    """

    n_pre, n_post = graph.shape
    if n_pre != n_post:
        raise ValueError("persistent SNN physical reorder requires an N x N graph.")
    if region_size <= 0:
        raise ValueError("region_size must be positive.")
    if post_tile_size <= 0:
        raise ValueError("post_tile_size must be positive.")
    supported = (
        "identity",
        "random",
        "global_fanout",
        "local_fanout",
        "global_primary_tile",
        "local_primary_tile",
    )
    if method not in supported:
        raise ValueError(f"Unknown reorder method: {method}.")

    _validate_graph(graph, n_pre)
    _validate_graph_indices(graph.indices, n_post)
    new_to_old = _build_new_to_old(
        graph,
        method,
        region_size,
        post_tile_size,
        seed,
    )
    old_to_new = torch.empty_like(new_to_old)
    old_to_new[new_to_old] = torch.arange(
        n_pre,
        device=new_to_old.device,
        dtype=new_to_old.dtype,
    )

    old_indptr = graph.indptr.to(torch.long)
    old_degrees = old_indptr[1:] - old_indptr[:-1]
    new_degrees = old_degrees.index_select(0, new_to_old)
    new_indptr_long = torch.zeros(
        n_pre + 1,
        device=graph.indptr.device,
        dtype=torch.long,
    )
    new_indptr_long[1:] = torch.cumsum(new_degrees, dim=0)
    edge_count = graph.indices.numel()
    new_edge_rows = torch.repeat_interleave(
        torch.arange(n_pre, device=graph.indptr.device),
        new_degrees,
    )
    new_row_starts = torch.repeat_interleave(
        new_indptr_long[:-1],
        new_degrees,
    )
    relative_edges = (
        torch.arange(edge_count, device=graph.indptr.device) - new_row_starts
    )
    old_edge_rows = new_to_old.index_select(0, new_edge_rows)
    old_edges = old_indptr.index_select(0, old_edge_rows) + relative_edges
    old_posts = graph.indices.to(torch.long).index_select(0, old_edges)
    new_indices = old_to_new.index_select(0, old_posts)
    new_weights = graph.weight.index_select(0, old_edges)
    new_delay = (
        graph.delay.index_select(0, old_edges) if graph.delay is not None else None
    )

    if sort_row_edges and edge_count > 0:
        edge_keys = new_edge_rows * n_pre + new_indices
        edge_order = torch.argsort(edge_keys, stable=True)
        new_indices = new_indices.index_select(0, edge_order)
        new_weights = new_weights.index_select(0, edge_order)
        if new_delay is not None:
            new_delay = new_delay.index_select(0, edge_order)

    reordered_graph = EventCSRGraph(
        indptr=new_indptr_long.to(graph.indptr.dtype).contiguous(),
        indices=new_indices.to(graph.indices.dtype).contiguous(),
        weight=new_weights.contiguous(),
        delay=new_delay.contiguous() if new_delay is not None else None,
        shape=graph.shape,
    )
    return PersistentSNNReorderPlan(
        new_to_old=new_to_old,
        old_to_new=old_to_new,
        reordered_graph=reordered_graph,
        method=method,
        region_size=region_size,
        post_tile_size=post_tile_size,
        sort_row_edges=sort_row_edges,
    )


def reorder_persistent_snn_state(
    state: PersistentSNNState,
    plan: PersistentSNNReorderPlan,
) -> PersistentSNNState:
    """Map caller state from original IDs to physical IDs."""

    permutation = plan.new_to_old

    def reorder(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return tensor.index_select(-1, permutation) if tensor is not None else None

    return PersistentSNNState(
        v=state.v.index_select(-1, permutation),
        psc=state.psc.index_select(-1, permutation),
        refractory=reorder(state.refractory),
        delay_ring=reorder(state.delay_ring),
    )


def restore_persistent_snn_state(
    state: PersistentSNNState,
    plan: PersistentSNNReorderPlan,
) -> PersistentSNNState:
    """Map physical state back to the caller's original IDs."""

    permutation = plan.old_to_new

    def restore(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return tensor.index_select(-1, permutation) if tensor is not None else None

    return PersistentSNNState(
        v=state.v.index_select(-1, permutation),
        psc=state.psc.index_select(-1, permutation),
        refractory=restore(state.refractory),
        delay_ring=restore(state.delay_ring),
    )


def reorder_windowed_spike_events(
    events: WindowedSpikeEvents,
    plan: PersistentSNNReorderPlan,
) -> WindowedSpikeEvents:
    """Map sparse input event indices from original IDs to physical IDs."""

    indices = plan.old_to_new.index_select(0, events.indices.to(torch.long))
    return WindowedSpikeEvents(
        offsets=events.offsets,
        indices=indices.to(events.indices.dtype).contiguous(),
        values=events.values,
        shape=events.shape,
    )


def restore_persistent_snn_output(
    output: PersistentSNNOutput,
    plan: PersistentSNNReorderPlan,
) -> PersistentSNNOutput:
    """Map dense/event output and state back to original neuron IDs."""

    spikes = (
        output.spikes.index_select(-1, plan.old_to_new)
        if output.spikes is not None
        else None
    )
    spike_events = None
    if output.spike_events is not None:
        restored_indices = plan.new_to_old.index_select(
            0,
            output.spike_events.indices.to(torch.long),
        )
        spike_events = WindowedSpikeEvents(
            offsets=output.spike_events.offsets,
            indices=restored_indices.to(output.spike_events.indices.dtype),
            values=output.spike_events.values,
            shape=output.spike_events.shape,
        )
    return PersistentSNNOutput(
        spikes=spikes,
        spike_events=spike_events,
        state=restore_persistent_snn_state(output.state, plan),
    )


def torch_stub_persistent_snn_forward(
    events: WindowedSpikeEvents,
    graph: EventCSRGraph,
    state: PersistentSNNState,
    params: PersistentSNNParams | None = None,
    *,
    return_mode: ReturnMode = "dense",
) -> PersistentSNNOutput:
    """Run the no-op persistent SNN reference implementation.

    The stub proves the operator contract without doing neural dynamics:
    state tensors are returned unchanged, dense spikes are all zero, and event
    output contains no spikes. It is intentionally deterministic and device
    preserving, so benchmarks can be used before the CUDA kernel exists.
    """

    del params
    t_steps, batch_size, n_pre = _validate_events(events)
    n_post = _validate_graph(graph, n_pre)
    _validate_state(state, batch_size=batch_size, n_post=n_post)

    dense_spikes = None
    if return_mode in ("dense", "both"):
        dense_spikes = torch.zeros(
            (t_steps, batch_size, n_post),
            device=state.v.device,
            dtype=state.v.dtype,
        )

    spike_events = None
    if return_mode in ("events", "both"):
        event_offsets = torch.zeros_like(events.offsets)
        event_indices = torch.empty(
            (0,), device=events.indices.device, dtype=events.indices.dtype
        )
        spike_events = WindowedSpikeEvents(
            offsets=event_offsets,
            indices=event_indices,
            values=None,
            shape=(t_steps, batch_size, n_post),
        )

    return PersistentSNNOutput(
        spikes=dense_spikes,
        spike_events=spike_events,
        state=state,
    )


def _has_cuda_op() -> bool:
    return hasattr(torch.ops, "btorch_cuda") and hasattr(
        torch.ops.btorch_cuda, "persistent_snn_forward"
    )


def _has_cuda_binned_op() -> bool:
    return hasattr(torch.ops, "btorch_cuda") and hasattr(
        torch.ops.btorch_cuda, "persistent_snn_forward_binned"
    )


def _has_cuda_spike_block_op() -> bool:
    return hasattr(torch.ops, "btorch_cuda") and hasattr(
        torch.ops.btorch_cuda, "persistent_snn_forward_spike_block"
    )


def _ensure_cuda_op(
    *,
    fanout_binning: bool = False,
    spike_block: bool = False,
) -> None:
    if spike_block:
        has_required_op = _has_cuda_spike_block_op()
    elif fanout_binning:
        has_required_op = _has_cuda_binned_op()
    else:
        has_required_op = _has_cuda_op()
    if has_required_op:
        return
    from .persistent import plain_version

    plain_version.load()


def _make_high_fanout_mask(graph: EventCSRGraph) -> torch.Tensor:
    """Create int32 high-fanout metadata for the binned CUDA kernel."""

    high_fanout = (
        (graph.indptr[1:] - graph.indptr[:-1]) >= _FANOUT_BINNING_THRESHOLD
    ).to(torch.int32)
    return high_fanout.contiguous()


# Delay tensors already confirmed all-zero, keyed by id(). `graph.delay` is part
# of a graph's fixed structure and is typically reused unchanged across many
# forward() calls (e.g. sequential windows of one long sequence) -- caching this
# avoids re-running an O(E) reduction + device->host sync on every single call
# for a value that never changes.
#
# Keyed by id() rather than stored in a `weakref.WeakSet`/`WeakKeyDictionary`:
# those containers fall back to `==` on hash-bucket lookups, and torch.Tensor's
# `__eq__` is elementwise (returns a Tensor, not a bool), which raises
# "Boolean value of Tensor with more than one value is ambiguous" -- even when
# comparing a tensor against itself, since weakref's own `__eq__` has no
# identity shortcut. Each entry carries a weakref-with-callback so it's evicted
# the moment the tensor is actually garbage collected, which also avoids the
# id()-reuse hazard (a new, unrelated tensor later allocated at the same
# address would otherwise appear to hit the cache).
_zero_delay_cache: dict[int, "weakref.ref[torch.Tensor]"] = {}


# Fanout metadata belongs to the graph structure and is reused across windows.
# Keep it alive only while the corresponding indptr tensor is alive, and
# invalidate it after any in-place structural mutation.
_high_fanout_cache: dict[
    int, tuple["weakref.ref[torch.Tensor]", int, torch.Tensor]
] = {}


def _delay_confirmed_zero(delay: torch.Tensor) -> bool:
    ref = _zero_delay_cache.get(id(delay))
    return ref is not None and ref() is delay


def _remember_delay_confirmed_zero(delay: torch.Tensor) -> None:
    key = id(delay)

    def _evict(_: object, key: int = key) -> None:
        _zero_delay_cache.pop(key, None)

    _zero_delay_cache[key] = weakref.ref(delay, _evict)


def _cached_high_fanout_mask(graph: EventCSRGraph) -> torch.Tensor:
    indptr = graph.indptr
    cached = _high_fanout_cache.get(id(indptr))
    if (
        cached is not None
        and cached[0]() is indptr
        and cached[1] == indptr._version
    ):
        return cached[2]

    high_fanout = _make_high_fanout_mask(graph)
    key = id(indptr)

    def _evict(_: object, key: int = key) -> None:
        _high_fanout_cache.pop(key, None)

    _high_fanout_cache[key] = (
        weakref.ref(indptr, _evict),
        indptr._version,
        high_fanout,
    )
    return high_fanout


def _cuda_persistent_snn_forward(
    events: WindowedSpikeEvents,
    graph: EventCSRGraph,
    state: PersistentSNNState,
    params: PersistentSNNParams,
    *,
    return_mode: ReturnMode,
    fanout_binning: bool,
    spike_block: bool,
    workspace: PersistentSNNWorkspace | None,
) -> PersistentSNNOutput:
    t_steps, batch_size, n_pre = _validate_events(events)
    n_post = _validate_graph(graph, n_pre)
    _validate_state(state, batch_size=batch_size, n_post=n_post)
    if n_pre != n_post:
        raise ValueError("cuda_persistent v1 requires a recurrent N x N graph.")
    if params.hard_reset:
        raise ValueError("cuda_persistent v1 only supports hard_reset=False.")
    if state.refractory is not None:
        raise ValueError("cuda_persistent v1 does not support refractory state.")
    if state.delay_ring is not None:
        raise ValueError("cuda_persistent v1 does not support delay_ring state.")
    if events.offsets.dtype != torch.int32 or events.indices.dtype != torch.int32:
        raise TypeError("cuda_persistent v1 requires int32 event tensors.")
    if graph.indptr.dtype != torch.int32 or graph.indices.dtype != torch.int32:
        raise TypeError("cuda_persistent v1 requires int32 graph indices.")
    _validate_graph_indices(graph.indices, n_post)
    if graph.delay is not None and graph.delay.dtype != torch.int32:
        raise TypeError("cuda_persistent v1 requires int32 graph delay.")
    delay_validated = False
    if graph.delay is not None:
        if not _delay_confirmed_zero(graph.delay):
            if torch.count_nonzero(graph.delay).item() != 0:
                raise ValueError("cuda_persistent v1 does not support nonzero delay.")
            _remember_delay_confirmed_zero(graph.delay)
        delay_validated = True
    if state.v.dtype != torch.float32 or state.psc.dtype != torch.float32:
        raise TypeError("cuda_persistent v1 requires float32 state tensors.")
    if graph.weight.dtype != torch.float32:
        raise TypeError("cuda_persistent v1 requires float32 graph weights.")
    if events.values is not None and events.values.dtype != torch.float32:
        raise TypeError("cuda_persistent v1 requires float32 event values.")

    if fanout_binning and spike_block:
        raise ValueError("fanout_binning and spike_block are mutually exclusive.")

    if workspace is None:
        workspace = make_persistent_snn_workspace(graph, batch_size)

    _ensure_cuda_op(
        fanout_binning=fanout_binning,
        spike_block=spike_block,
    )
    event_values = (
        events.values
        if events.values is not None
        else torch.empty((0,), device=events.indices.device, dtype=state.v.dtype)
    )
    graph_delay = (
        graph.delay
        if graph.delay is not None
        else torch.empty((0,), device=graph.indices.device, dtype=graph.indices.dtype)
    )
    return_events = return_mode in ("events", "both")
    return_dense = return_mode in ("dense", "both")
    op_args = (
        events.offsets,
        events.indices,
        event_values,
        events.values is not None,
        graph.indptr,
        graph.indices,
        graph.weight,
        True,
        workspace.input_current,
        workspace.queue_batch,
        workspace.queue_edge_start,
        workspace.queue_edge_end,
        workspace.spike_count,
        workspace.work_counter,
    )
    op_tail = (
        graph_delay,
        graph.delay is not None,
        delay_validated,
        state.v,
        state.psc,
        float(params.dt),
        float(params.tau_mem),
        float(params.tau_syn),
        float(params.v_threshold),
        float(params.v_reset),
        float(params.c_m),
        bool(params.hard_reset),
        return_dense,
        return_events,
    )
    if spike_block:
        (
            dense_spikes,
            event_offsets,
            event_indices,
            v_out,
            psc_out,
            _overflow,
        ) = torch.ops.btorch_cuda.persistent_snn_forward_spike_block(
            *op_args,
            *op_tail,
        )
    elif fanout_binning:
        graph_high_fanout = _cached_high_fanout_mask(graph)
        (
            dense_spikes,
            event_offsets,
            event_indices,
            v_out,
            psc_out,
            _overflow,
        ) = torch.ops.btorch_cuda.persistent_snn_forward_binned(
            *op_args,
            graph_high_fanout,
            *op_tail,
        )
    else:
        (
            dense_spikes,
            event_offsets,
            event_indices,
            v_out,
            psc_out,
            _overflow,
        ) = torch.ops.btorch_cuda.persistent_snn_forward(*op_args, *op_tail)

    spikes = dense_spikes if return_mode in ("dense", "both") else None
    spike_events = None
    if return_events:
        spike_events = WindowedSpikeEvents(
            offsets=event_offsets,
            indices=event_indices,
            values=None,
            shape=(t_steps, batch_size, n_post),
        )
    return PersistentSNNOutput(
        spikes=spikes,
        spike_events=spike_events,
        state=PersistentSNNState(v=v_out, psc=psc_out),
    )


def persistent_snn_forward(
    events: WindowedSpikeEvents,
    graph: EventCSRGraph,
    state: PersistentSNNState,
    params: PersistentSNNParams | None = None,
    *,
    backend: Backend = "auto",
    return_mode: ReturnMode = "dense",
    fanout_binning: bool = False,
    spike_block: bool = False,
    workspace: PersistentSNNWorkspace | None = None,
) -> PersistentSNNOutput:
    """Dispatch the persistent SNN operator.

    Args:
        events: Windowed input spike events.
        graph: CSR connectivity graph.
        state: Persistent state tensors.
        params: Scalar simulation parameters.
        backend: ``"torch_stub"`` for the reference no-op, ``"cuda_persistent"``
            for the future compiled operator, or ``"auto"`` to use CUDA when
            registered and fall back to the stub otherwise.
        return_mode: Select dense spikes, event spikes, or both.
        fanout_binning: Bucket recurrent work by fanout. Long rows are split
            into bounded warp tasks, while ordinary rows use 8-lane subwarps.
        spike_block: Group fired cells from each contiguous 32-neuron block
            into one task. Rows up to 16 edges use their relative block lane;
            fanout of at least 256 edges uses 1024-edge segments.
        workspace: Reusable CUDA scratch storage. Create it once with
            :func:`make_persistent_snn_workspace` for steady-state windows, or
            leave it unset to allocate scratch tensors for this call.

    Returns:
        Operator output with spikes/events and final state.
    """

    params = params or PersistentSNNParams()
    if backend not in ("auto", "torch_stub", "cuda_persistent"):
        raise ValueError(f"Unknown backend: {backend}.")

    if fanout_binning and spike_block:
        raise ValueError("fanout_binning and spike_block are mutually exclusive.")
    if spike_block:
        has_selected_cuda_op = _has_cuda_spike_block_op()
    elif fanout_binning:
        has_selected_cuda_op = _has_cuda_binned_op()
    else:
        has_selected_cuda_op = _has_cuda_op()
    use_cuda = backend == "cuda_persistent" or (
        backend == "auto" and has_selected_cuda_op
    )
    if use_cuda:
        return _cuda_persistent_snn_forward(
            events,
            graph,
            state,
            params,
            return_mode=return_mode,
            fanout_binning=fanout_binning,
            spike_block=spike_block,
            workspace=workspace,
        )

    return torch_stub_persistent_snn_forward(
        events,
        graph,
        state,
        params,
        return_mode=return_mode,
    )


__all__ = [
    "EventCSRGraph",
    "PersistentSNNOutput",
    "PersistentSNNParams",
    "PersistentSNNReorderPlan",
    "PersistentSNNState",
    "PersistentSNNWorkspace",
    "WindowedSpikeEvents",
    "build_persistent_snn_reorder_plan",
    "make_empty_state",
    "make_persistent_snn_workspace",
    "persistent_snn_forward",
    "reorder_persistent_snn_state",
    "reorder_windowed_spike_events",
    "restore_persistent_snn_output",
    "restore_persistent_snn_state",
    "torch_stub_persistent_snn_forward",
]
