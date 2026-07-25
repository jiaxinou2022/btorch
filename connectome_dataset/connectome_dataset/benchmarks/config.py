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
class SpMSpVDefaults:
    """y = A·x, x sparse (VDHA / event-driven spike propagation).

    ``vector_sparsities`` = fraction of x's entries that are nonzero, the VDHA paper's
    sweep (0.01, 0.05, 0.10, 0.20).
    """
    vector_sparsities: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    seed: int = 0
    warmup: int = 25
    rep: int = 100


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
class RSNNDeliveryDefaults:
    """Event-driven spike-delivery track (VDHA / MH-SpGEMM as the beNNch deliver phase).

    Spikes are driven at ``firing_rate_hz`` over ``timesteps`` steps of ``dt_ms``; ``batch``
    is the number of concurrent spike vectors (the SpGEMM operand width; 1 for SpMSpV).
    """
    firing_rate_hz: float = 15.0
    timesteps: int = 1000
    dt_ms: float = 1.0
    batch: int = 32  # >= MH-SpGEMM's BLOCK_SIZE; its B-tiling faults on very few columns
    seed: int = 0


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


@dataclass(frozen=True, slots=True)
class FlyBrainDefaults:
    """Shiu et al. (2024) whole-brain Drosophila LIF model (torch/btorch/flybrain.py).

    Sizes are configurable; test defaults are short (the paper uses t_run=1000 ms, n_run=30).
    """
    dt_ms: float = 0.1
    t_run_ms: float = 200.0
    n_run: int = 4
    r_poi_hz: float = 150.0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class MultiAreaDefaults:
    """Macaque multi-area model + cortical microcircuit (Schmidt et al. 2018).

    The neuron-level network is instantiated at a configurable scale (``n_scaling`` scales
    population sizes, ``k_scaling`` the in-degrees) — full scale is 4.13M neurons / ~24e9
    synapses, so every default here is deliberately SMALL and full scale is opt-in.
    Shared by the btorch (torch/btorch/microcircuit.py) and NEST (nest/) implementations.
    """
    microcircuit_area: str = "V1"
    microcircuit_scaling: float = 0.1   # single cortical area (~20k neurons at 0.1x)
    multiarea_scaling: float = 0.01     # all 32 areas (~41k neurons at 0.01x)
    k_scaling: float = 1.0
    dt_ms: float = 0.1
    timesteps: int = 1000               # 0.1 ms dt -> 100 ms of biological time
    warmup_ms: float = 100.0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class MouseV1Defaults:
    """Billeh GLIF mouse-V1 model (torch/btorch/mouse_v1.py).

    ``n_neurons`` sub-samples the r<400um core (full core 51,978) — configurable size, as the
    upstream ``load_sparse.py``; the default is small so the benchmark stays quick.
    """
    n_neurons: int = 8000        # mice_v1_guozhang (Billeh core): radial-core subsample size
    connected_selection: bool = False
    column_replicate: int = 1    # mice_column_v1: block-diagonal tiling of the 4,166 column
    dt_ms: float = 1.0
    timesteps: int = 1000
    warmup: int = 300
    bg_rate_hz: float = 250.0
    bg_weight_pa: float = 100.0
    seed: int = 0


SPMV_DEFAULTS = SpmvDefaults()
SPGEMM_DEFAULTS = SpgemmDefaults()
SPMSPV_DEFAULTS = SpMSpVDefaults()
RSNN_DEFAULTS = RSNNDefaults()
RSNN_DELIVERY_DEFAULTS = RSNNDeliveryDefaults()
BENNCH_DEFAULTS = BennchDefaults()
FLYBRAIN_DEFAULTS = FlyBrainDefaults()
MULTIAREA_DEFAULTS = MultiAreaDefaults()
MOUSE_V1_DEFAULTS = MouseV1Defaults()
