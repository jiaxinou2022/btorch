from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpmvDefaults:
    batch_sizes: tuple[int, ...] = (1, 8, 32, 128)
    warmup: int = 25
    rep: int = 100


@dataclass(frozen=True, slots=True)
class SpgemmDefaults:
    """C = A @ A.T (two-hop / common-neighbour / Markov-clustering expansion)."""
    warmup: int = 10
    rep: int = 50


@dataclass(frozen=True, slots=True)
class RSNNDefaults:
    timesteps: int = 50
    batch_size: int = 4
    num_input: int = 64
    num_output: int = 10
    dense_sizes: tuple[int, ...] = (64, 256, 512, 1024, 2048, 4096)
    warmup: int = 10
    rep: int = 50


@dataclass(frozen=True, slots=True)
class BennchDefaults:
    """Defaults for the beNNch balanced E-I track (rsnn/models.py, rsnn/phases.py)."""
    models: tuple[str, ...] = ("cuba", "coba")
    scales: tuple[float, ...] = (1.0,)  # strong scaling: sweep via --rsnn-scale
    timesteps: int = 1000  # 0.1 ms dt -> 100 ms of biological time
    seed: int = 0
    phase_reps: int = 5  # repetitions for the per-phase microbenchmarks
    warmup: int = 3
    rep: int = 20


SPMV_DEFAULTS = SpmvDefaults()
SPGEMM_DEFAULTS = SpgemmDefaults()
RSNN_DEFAULTS = RSNNDefaults()
BENNCH_DEFAULTS = BennchDefaults()
