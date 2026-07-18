# RSNN benchmark track (Phase 3) — beNNch methodology

This track benchmarks end-to-end recurrent spiking networks (RSNN): the inner loop is
a recurrent, fixed-operator sparse mat-vec over many timesteps (the connectome adjacency
reused every step). It adopts the **beNNch** benchmarking methodology and uses the
standard balanced excitatory-inhibitory benchmark networks as workloads.

- **beNNch**: Albers et al. (2022), *A modular workflow for performance benchmarking of
  neuronal network simulations*, Front. Neuroinform.
- **Benchmark networks**: the Vogels-Abbott / Brette-2007 balanced E-I networks, adapted
  from brainevent's `examples/COBA_2005.py` and `CUBA_2005.py`.

## Layout

| File | Role |
|------|------|
| `benchmarks/rsnn/models.py` | benchmark network models (`coba`, `cuba`, `connectome`) + `NetworkSpec` |
| `benchmarks/rsnn/phases.py` | beNNch phase-resolved measurement (build/compile, reference run, per-phase timing) |
| `benchmarks/jax/brainstate/test_einet.py` | pytest-benchmark leaf emitting records via `templates.base_extra_info` |
| `benchmarks/config.py::BennchDefaults` | models / scales / timesteps / seed / reps |
| `benchmarks/history/ingest.py` | derives spikes/s, synaptic-events/s, per-phase timings for rsnn records |

The identity tuple is unchanged: `framework=jax`, `provider=brainstate`,
`target=rsnn`, `variant=<model>` (`coba` / `cuba` / `connectome`). Records land on the
existing leaderboard, whose primary metric for `rsnn` is the **real-time factor** (lower
is better). The workload cell keys on `timesteps`, `batch_size`, `pass` (see
`history/ingest.py::_canonical_problem`), so different scales rank in their own cells
(the `case_id` embeds model + scale + neuron count).

## The models

All three are built from `brainpy.state` / `brainstate.nn` primitives — LIF-with-refractory
neurons, exponential synapses, event-driven connectivity — so this repo does not
reimplement the simulator, only the network wiring.

- **`cuba`** — current-based balanced network (V_rest −49 mV, w_exc 1.62 mS, w_inh −9.0 mS).
- **`coba`** — conductance-based balanced network (V_rest −60 mV, w_exc 0.6 mS,
  w_inh 6.7 mS, reversal potentials 0 / −80 mV).
- **`connectome`** — the repo's raison d'être: a LIF population whose recurrent operator
  is a **connectome adjacency CSR** (delivered through `brainstate.nn.SparseLinear`),
  fed from the catalog (`--graph`).

`coba`/`cuba` size by `scale`: `n = 3200·scale` excitatory + `800·scale` inhibitory,
each pre-neuron making a fixed 80 post connections (event-driven fixed-probability). This
gives clean **strong scaling** for a fixed connectome and **weak scaling** across scales.

## Phase-resolved measurement (beNNch)

beNNch separates network **construction** from **state propagation**, and decomposes one
propagation step into `update` (neuron dynamics), `gather` (spike collocation),
`communicate` (inter-rank spike exchange) and `deliver` (synaptic delivery). Mapping onto
a clock-driven, single-device, XLA-fused simulation:

| beNNch phase | Here | Recorded as |
|--------------|------|-------------|
| build/construct | network construction + state init + JIT compile | `build_ms`, `compile_ms` |
| update | LIF integrate / threshold / reset (`_neuron`) | `phase_update_ms` |
| gather | spike readout (`get_spike`) — trivial, folded into update | `phase_gather_ms` = 0 |
| communicate | none (single device, no MPI) | `phase_communicate_ms` = 0 |
| deliver | event-driven SpMV + synapse filtering (`_deliver`) | `phase_deliver_ms` |

XLA fuses the full step, so the per-phase costs cannot be carved out of the fused kernel;
instead `update` and `deliver` are timed as **isolated `timesteps`-long sub-loops** on
fresh network instances (re-jitting one stateful net into several interleaved loops leaks
tracers). The `deliver` sub-loop is driven by a fixed random spike vector at the measured
firing probability, so the event-driven cost is exercised at the network's operating
point. `phase_full_ms` is the full step timed the same way, for reference. This is an
approximate decomposition — the honest, framework-limited reading of beNNch's phases.

The **primary** hot-loop timing is done by pytest-benchmark (median over rounds), from
which ingest derives the real-time factor `T_wall / T_model` (`T_model = timesteps · dt`,
`dt = 0.1 ms`). A separate `reference_run` (fresh state) counts spikes and synaptic
events (spikes × out-degree), from which ingest computes **spikes/s** and
**synaptic-events/s**; it also reports the mean **firing rate** (Hz) to validate the
network is in the balanced regime (~a few tens of Hz at `scale=1`).

## Metrics recorded

- `real_time_factor` (primary, asc) — `T_wall / T_model`; `<1` is faster-than-real-time.
- `spikes_per_s`, `synaptic_events_per_s` — beNNch throughput rates.
- `firing_rate_hz` — steady-state mean rate (validation / regime check).
- `phase_{update,deliver,gather,communicate}_ms`, `phase_full_ms`, `build_ms`,
  `compile_ms` — the per-phase wall-time breakdown (beNNch "flip-plot" inputs).
- `tflops` — from `flops_rsnn_step`.

## Running

```bash
# CPU smoke (small scale, few timesteps)
pytest connectome_dataset/benchmarks/jax/brainstate/ \
    --rsnn-model cuba,coba --rsnn-scale 0.05 --rsnn-timesteps 200 --warmup 1 --rounds 2

# fast CPU validation (assertions only, no benchmark loop)
pytest connectome_dataset/benchmarks/jax/brainstate/test_einet.py::test_bennch_metrics_sane

# GPU strong-scaling sweep + ingest + report
sbatch scripts/bench_rsnn.sbatch                     # scales 1,2,4,8,16
sbatch scripts/bench_rsnn.sbatch --rsnn-scale 1,2    # smaller
```

CLI options (in `benchmarks/conftest.py`): `--rsnn-model coba|cuba|connectome` (CSV),
`--rsnn-scale` (CSV), `--rsnn-timesteps`. Defaults live in `config.py::BennchDefaults`.

## Dependencies

The models need the full brain* simulator stack (`brainstate`, `brainpy`, `braintools`,
`brainunit`, `brainevent`). The leaf `pytest.importorskip`s them, so the track skips
cleanly where they are absent. As of this writing all are importable in `ml-py312`.

## Notes / follow-ups

- `docs/benchmarks.md` (the practical guide, owned elsewhere) should gain a short RSNN
  section: the three models, the `--rsnn-*` options, and that `rsnn` ranks on
  `real_time_factor`. Not edited here to avoid a merge conflict.
- Seeds: a single seed (`BennchDefaults.seed`) is used per run; sweeping seeds for
  robustness is a matter of repeated invocations (seed is not part of the cell key).
- Not yet exercised on GPU (sm_120): submit `scripts/bench_rsnn.sbatch`. brainevent's
  event kernels build for the node's Blackwell arch via the same conda toolchain patch
  the SpMV JAX suite uses (`benchmarks/backend_utils.py`, `jax/conftest.py`).
- PERKS persistent-kernel / CUDA-graph comparison for the recurrent operator (design.md
  §4b) remains future work on top of this track.
