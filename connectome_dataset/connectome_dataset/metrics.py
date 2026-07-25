"""FLOP counting helpers for benchmark problem-size annotations."""
from __future__ import annotations


def flops_spmv(nnz: int) -> int:
    """2 * nnz FLOPs (one mul + one add per nonzero)."""
    return 2 * nnz


def flops_spmm(nnz: int, batch: int) -> int:
    """2 * nnz * batch FLOPs."""
    return 2 * nnz * batch


def flops_spmspv(products: int) -> int:
    """2 * (scalar multiply-adds) for y = A·x, x sparse.

    ``products`` = sum over active x entries of the nnz in the corresponding column of A
    (the work actually touched), not 2·nnz(A).
    """
    return 2 * products


def flops_spgemm(intermediate_products: int) -> int:
    """2 * (intermediate multiply-adds) FLOPs for C = A @ B.

    ``intermediate_products`` is the number of scalar products formed before
    accumulation (a.k.a. the ``compression_ratio`` numerator), not nnz(C).
    """
    return 2 * intermediate_products


def flops_rsnn_step(nnz_recurrent: int, n_neurons: int) -> int:
    """Approximate FLOPs per timestep: SpMV + neuron update (~6 ops/neuron)."""
    return 2 * nnz_recurrent + 6 * n_neurons


def gflops(flops: int, ms: float) -> float:
    """GFLOP/s given total FLOPs and wall time in ms."""
    return flops / (ms * 1e-3) / 1e9


def tflops(flops: int, ms: float) -> float:
    """TFLOP/s given total FLOPs and wall time in ms."""
    return flops / (ms * 1e-3) / 1e12


def real_time_factor(wall_s: float, model_s: float) -> float:
    """Twall / Tmodel. <1 is faster-than-real-time (beNNch convention)."""
    return wall_s / model_s if model_s > 0 else float("inf")


def events_per_s(n_events: int, wall_s: float) -> float:
    """Spike or synaptic-event throughput."""
    return n_events / wall_s if wall_s > 0 else 0.0


def spikes_per_s(n_spikes: int, wall_s: float) -> float:
    """Spike throughput (beNNch spikes/s) over the timed hot loop."""
    return events_per_s(n_spikes, wall_s)


def synaptic_events_per_s(n_syn_events: int, wall_s: float) -> float:
    """Synaptic-event throughput (beNNch synaptic-events/s): spikes x out-degree."""
    return events_per_s(n_syn_events, wall_s)
