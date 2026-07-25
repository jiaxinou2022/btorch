# multiarea — macaque multi-area model + cortical microcircuit

The INM-6 **multi-area model** (Schmidt et al. 2018) is a layer-resolved spiking network
of macaque visual cortex: **32 areas × 8 populations (254 total), ~4.13M neurons, ~24
billion synapses** at full scale. This repo integrates it — and its single-area
Potjans-Diesmann-type **cortical microcircuit** — as configurable-scale benchmark graphs
and as RSNN models in **btorch** and **NEST**.

- Paper: Schmidt et al. (2018) PLOS Comput Biol 14(9): e1006359. doi:10.1371/journal.pcbi.1006359
- Upstream: <https://github.com/INM-6/multi-area-model>

## Everything is scale-configurable

Full scale (~24e9 synapses) cannot be instantiated at neuron level, so the model is kept
at the **mesoscale** — population sizes `N` and mean in-degrees `K` between the 254
populations — and `connectome_dataset.cortical_network` builds the neuron-level network at
any requested scale:

| knob | effect | where |
|------|--------|-------|
| `n_scaling` | scales population sizes | `config.MULTIAREA_DEFAULTS`, `CONNECTOME_MICROCIRCUIT_SCALING`, `CONNECTOME_MULTIAREA_SCALING` |
| `k_scaling` | scales in-degrees | same |
| `max_nnz` | hard cap; instantiation **raises** above it | `instantiate_connectivity(..., max_nnz=4e8)` |

Defaults are small (microcircuit `n_scaling=0.05` ≈ 10k neurons; multiarea `0.005` ≈ 20k),
so tests and routine catalog builds never create large networks. Full MAM at `n_scaling=1`
refuses:

```
MemoryError: instantiation would create 24,126,497,101 synapses (> max_nnz=400,000,000)
```

## Two views, one machinery

```python
from connectome_dataset.cortical_network import microcircuit_spec, multiarea_spec, instantiate_connectivity

mc = microcircuit_spec(area="V1", n_scaling=0.1)   # one area, 8 populations
W  = instantiate_connectivity(mc, seed=0)           # signed CSR (pA), rows=pre cols=post

mam = multiarea_spec(n_scaling=0.01)                # all 32 areas, 254 populations
```

Catalog graphs: `microcircuit_v1` and `multiarea_mam` (both scale-configurable via the env
vars above). Staging: run `scripts/build_multiarea_mesoscale.py` once, then `connectome-catalog`.

## Models (btorch + NEST)

`iaf_psc_exp` LIF neurons — C_m=250 pF, τ_m=10 ms, V_th=−50 mV, V_reset=E_L=−65 mV,
τ_syn=0.5 ms, t_ref=2 ms — with current-based exponential synapses and a per-population
Poisson background. Since `E_L == V_reset`, this maps exactly onto btorch
`LIF(v_reset=−65, tau=τ_m, c_m=C_m)` driven by the synaptic current.

- **btorch** — `torch/btorch/microcircuit.py` (`CorticalMicrocircuitModel`, `LIF` + `SparseConn`),
  recorded on the RSNN track as `torch.btorch.rsnn.{microcircuit,multiarea}`.
- **NEST** — `benchmarks/nest/native/microcircuit.py`, built the native way with
  `fixed_indegree` per population pair — an **independent reference**, recorded as
  `nest.native.rsnn.{microcircuit,multiarea}`. NEST is a new benchmark *framework* (peer of
  jax/torch), wired into `run_all.py` and skipped cleanly where NEST is absent.

### Validation

Both implementations reproduce the model's asynchronous-irregular ground state: L2/3E ≈ 0.1
Hz, inhibitory > excitatory, a few Hz mean. Same-scale btorch-vs-NEST **per-population rate
correlation r ≈ 0.94** (independent frameworks and independent random draws), e.g. for V1:

| pop | btorch | NEST |
|-----|--------|------|
| V1_23E | 0.12 | 0.09 |
| V1_23I | 1.78 | 1.57 |
| V1_4E  | 2.93 | 1.92 |
| V1_5I  | 5.31 | 4.34 |
| V1_6I  | 4.85 | 4.48 |

The small offsets are expected: btorch uses forward-Euler membrane integration and a single
mean recurrent delay, NEST uses exact integration and per-edge E/I delays, and the two draw
different random networks.
