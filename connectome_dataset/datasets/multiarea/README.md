# multiarea

Repo-local metadata for the macaque **multi-area model** (Schmidt et al. 2018) — the
INM-6 layer-resolved spiking network of macaque visual cortex — and its single-area
**cortical microcircuit** (Potjans-Diesmann-type). Both are integrated as configurable-scale
benchmark graphs (`microcircuit_v1`, `multiarea_mam`) and as RSNN models in **btorch** and
**NEST**.

## Why it is scale-configurable (the whole point)

Full scale is **4,130,056 neurons and ~24 billion synapses** — impossible to instantiate
at neuron level. The model is therefore stored at the **mesoscale**: population sizes `N`
and mean in-degrees `K` between the 254 populations (32 areas × 8 layers/types). From that
tiny description, `connectome_dataset.cortical_network` instantiates the neuron-level
network at **any** scale:

- `n_scaling` scales population sizes, `k_scaling` scales in-degrees.
- `instantiate_connectivity(..., max_nnz=...)` **refuses** to build more than a cap
  (default 400M synapses), so an over-ambitious scale fails fast instead of OOM-ing.

Test/catalog defaults are deliberately small (`config.MULTIAREA_DEFAULTS`; catalog reads
`CONNECTOME_MICROCIRCUIT_SCALING` / `CONNECTOME_MULTIAREA_SCALING`). Opt into larger scales
via those knobs; full scale is never built automatically.

## Source (staged mesoscale, not the neuron-level network)

Extracted once from the `multi-area-model` checkout by
[`scripts/build_multiarea_mesoscale.py`](../../scripts/build_multiarea_mesoscale.py) into
`data/external/multiarea_mesoscale/` (DVC-managed, not committed):

| File | Contents |
|------|----------|
| `multiarea_mesoscale.npz` | `N` (254,), `K_int` (254×254), `K_ext` (254,), `W_int` (254×254, pA, signed), `W_ext` (254,), `labels` |
| `params.json` | `iaf_psc_exp` neuron params, synaptic delays, background rate |

```bash
micromamba run -n ml-py312 python scripts/build_multiarea_mesoscale.py \
    --src /home/fanqixuan/src/multi-area-model
micromamba run -n ml-py312 connectome-catalog        # registers microcircuit_v1 + multiarea_mam
```

## Models

`iaf_psc_exp` LIF neurons (C_m=250 pF, τ_m=10 ms, V_th=−50, V_reset=E_L=−65, τ_syn=0.5 ms,
t_ref=2 ms), current-based exponential synapses, Poisson background:

- **btorch** — `torch/btorch/microcircuit.py` (`CorticalMicrocircuitModel`, LIF + `SparseConn`).
- **NEST** — `benchmarks/nest/native/microcircuit.py` (native `fixed_indegree`), the reference.

The two agree on the characteristic asynchronous-irregular ground state (per-population rate
correlation **r ≈ 0.94**; L2/3E ≈ 0.1 Hz, inhibitory > excitatory, a few Hz mean). See
[docs/multiarea.md](../../docs/multiarea.md).

## Reference

Schmidt M, Bakker R, Shen K, Bezgin G, Diesmann M, van Albada SJ (2018) *A multi-scale
layer-resolved spiking network model of resting-state dynamics in macaque cortex.* PLOS
Comput Biol 14(9): e1006359. doi:10.1371/journal.pcbi.1006359
