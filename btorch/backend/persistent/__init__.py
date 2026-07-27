"""Persistent-kernel backend implementations."""

from .reorder import (
    NeuronPermutation,
    ReorderConfig,
    ReorderedPersistentInputs,
    build_neuron_permutation,
    dominant_post_blocks,
    prepare_reordered_inputs,
    reorder_events,
    reorder_graph,
    reorder_state,
    restore_output,
    restore_state,
)


__all__ = [
    "NeuronPermutation",
    "ReorderConfig",
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
