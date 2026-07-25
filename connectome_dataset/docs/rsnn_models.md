# RSNN models

Every recurrent spiking model in the benchmark suite, by family: neuron and synapse
dynamics, connectivity substrate, input drive, and validation. Identity is the tuple
**`framework · provider · target · variant`** (invoking runtime · library/algorithm ·
`rsnn` · the specific model).

Published biophysical models (fly, macaque, and two distinct mouse-V1 GLIF models) sit
alongside the original balanced / generic RSNN benchmarks. **01, 02, 04 are faithful published
models** (fly validated against brian2, macaque against NEST); **03 is the original kernel-timing
track**.

| Model | Neuron | Synapse | Connectivity | Size | Frameworks | Validation |
|-------|--------|---------|--------------|------|------------|------------|
| `flybrain` | LIF (v₀=v_rst) | exp, current, τ5 · d1.8 | FlyWire v783, signed, 40% inh | 138,639 / 15.1M | torch·btorch, brian2·native | r 0.999 |
| `microcircuit` / `multiarea` | iaf_psc_exp | exp, current, τ0.5 | mesoscale (N,K,W), 32 areas / 254 pops | configurable ≤ 4.13M | nest·native, torch·btorch | r 0.94 |
| `mouse_v1_guozhang` | GLIF3, per cell type (111) | exp, current, τ5 | Billeh V1 core (signed) | subsample ≤ 51,978 | torch·btorch | Billeh 2020 |
| `mouse_column_v1` | GLIF3, per cell type (21) | exp, current, τ5 | generated VISp column | 4,166 × replicate | torch·btorch | Allen VISp GLIF |
| `coba` / `cuba` / `connectome` | LIFRef | exp, COBA/CUBA, τ5/10 | EventFixedProb or CSR | 4000·scale | jax·brainstate | beNNch phases |
| `dense` / `sparse` | GLIF3 (2 ASC) | alpha (AlphaPSC) τ5 | Dense / Sparse (random on pattern) | 64→N→10 | torch·btorch | timing track |

---

## 01 · Whole-brain *Drosophila* — `flybrain`

Shiu et al., *Nature* 2024. `torch·btorch·rsnn·flybrain` (production) + `brian2·native·rsnn·flybrain`
(reference). See [flywire_783.md](flywire_783.md).

**Neuron — leaky integrate & fire**
```
dv/dt = (v₀ − v + g) / t_mbr
spike when v > v_th  →  v = v_rst, g = 0    (refractory t_rfc)
```
v₀ = v_rst = −52 mV · v_th = −45 mV · t_mbr = 20 ms · t_rfc = 2.2 ms · hard reset · dt = 0.1 ms

**Synapse — exponential, current-based**
```
dg/dt = −g / tau        (τ = 5 ms)
pre-spike (delay 1.8 ms):  g += w
w = (Excitatory × Connectivity) · w_syn      (w_syn = 0.275 mV, signed: inh < 0)
```
btorch op: `SparseConn`.

- **Connectome** — FlyWire v783: 138,639 neurons, 15.1M signed synapses, ~40% inhibitory;
  loaded from the original `Completeness_783.csv` + `Connectivity_783.parquet` (no npz).
- **Drive** — all 8 sensory types at once (sugar · sugar_left · bitter · Ir94e · water GRNs +
  Johnston's-Organ JON_CE / JON_F / JON_D_m) = 230 neurons, Poisson `r_poi = 150 Hz` (forced
  spikes). Silencing zeros a neuron's adjacency row + column.
- **Tonic background** (on by default) — independent per-neuron Poisson *into g* (diffusion
  regime, smooth), `3500 Hz × 0.30 mV` → ~8 Hz asynchronous-irregular baseline (vs 0.4 Hz
  stimulus-only).
- **v bounding, no clamp** — hard reset + masking synaptic input to refractory neurons
  (btorch integrates v every step; the Shiu equations are `(unless refractory)`) → no overshoot.
  Deep hyperpolarisation of strongly-inhibited neurons is faithful (brian2 shows it too).
- **Validation** — per-neuron rate **r = 0.999 vs brian2** (base dynamics).

---

## 02 · Macaque multi-area cortex — `microcircuit` / `multiarea`

Schmidt et al., *PLOS Comput Biol* 2018. `nest·native·rsnn·{microcircuit,multiarea}` (reference,
native `fixed_indegree`) + `torch·btorch·rsnn·{microcircuit,multiarea}`. See [multiarea.md](multiarea.md).

**Neuron — iaf_psc_exp (LIF)**
```
dV/dt = −(V − E_L)/tau_m + I/C_m
spike when V > V_th  →  V = V_reset    (refractory t_ref)
```
E_L = V_reset = −65 mV · V_th = −50 mV · C_m = 250 pF · tau_m = 10 ms · t_ref = 2 ms · tau_syn = 0.5 ms

**Connectivity — mesoscale → neurons**
```
per pair (target t ← source s):
  in-degree = round(K[t,s] · N[t])
  weight    = W[t,s]   (pA, signed)
scaled by n_scaling (population sizes) · k_scaling (in-degrees)
```
254 populations · 32 areas · delay E/I = 1.5 / 0.75 ms.

- **Two views** — `microcircuit` = one cortical area (V1, 8 populations); `multiarea` = all 32
  areas. Same machinery, different slice + scale.
- **Scale guard** — full scale is 4.13M neurons / ~24×10⁹ synapses, never instantiated;
  `instantiate_connectivity` refuses above `max_nnz` (4×10⁸). Test/catalog defaults are tiny
  (~10–40k neurons).
- **Drive** — per-population Poisson background, `rate_ext = 10 Hz × K_ext`, weight `W_ext`.
- **Validation** — per-population rate **r = 0.94, btorch vs NEST**; textbook ground state:
  L2/3E ≈ 0.1 Hz, inhibitory > excitatory, ~2–3 Hz mean.

---

## 03 · Balanced & generic RSNN — the original benchmark track

### `coba` / `cuba` / `connectome` — `jax·brainstate·rsnn·*`

beNNch balanced excitatory–inhibitory networks on brainstate/brainpy (Vogels-Abbott / Brette
2007), from brainevent's `COBA/CUBA_2005`. See `benchmarks/rsnn/models.py`.

- **Neuron** — `LIFRef`: V_th = −50 mV · V_reset = −60 mV · tau = 20 ms · tau_ref = 5 ms ·
  V_rest = −49 (cuba/connectome) / −60 (coba) · dt = 0.1 ms.
- **Synapse** — `Expon`: τ_E = 5 ms / τ_I = 10 ms. `cuba` CUBA (w_E 1.62 / w_I −9) ·
  `coba` COBA (E 0 / −80 mV; w_E 0.6 / w_I 6.7).
- **Connectivity** — `EventFixedProb`, out-degree 80 (E and I), `n_exc = 3200·scale`,
  `n_inh = 800·scale`; `connectome` uses `SparseLinear` (catalog CSR) as the recurrent operator.

### `DenseRSNN` / `SparseRSNN` — `torch·btorch·rsnn·{dense,sparse}`

Generic recurrent workload for sparse-kernel timing. See `benchmarks/torch/btorch/rsnn.py`.

- **Neuron** — `GLIF3` (two after-spike currents): v_threshold = −45 mV · v_reset = −60 mV ·
  c_m = 2.0 · tau = 20 ms · tau_ref = 2 ms · k = [0.1, 0.2] · asc_amps = [1, −2].
- **Synapse** — `AlphaPSC`, tau_syn = 5 ms.
- **Connectivity** — `DenseConn` (hidden 64…4096) or `SparseConn` (connectome pattern, random
  seeded weights); linear `fc_in` 64→N and `fc_out` N→10.
- The `native` / `vdha` / `mh_spgemm` spike-delivery variants also live under this leaf.

---

## 04 · Mouse V1 — two distinct Billeh-GLIF models

`mice_column_v1` and `mice_v1_guozhang` are **two different mouse-V1 models** (different source
repos, connectivity, and cell-type fits), not one network at two scales. Both are heterogeneous
**GLIF3** populations — every neuron takes its own cell type's parameter fit — sharing one btorch
class (`torch/btorch/mouse_v1.py`, `MouseV1GLIF`) with per-model loaders. The mouse analogue of
`flybrain`.

**Neuron — GLIF3, per cell type**
```
dV/dt = −(V − E_L)/tau + (I + Σ I_asc)/C_m       tau = C_m / g
dI_asc/dt = −k · I_asc      spike → V = V_reset, I_asc += asc_amps   (refractory t_ref)
```
Synapse: exponential current (τ_syn 5 ms), signed weights (pA) via `SparseConn`; per-neuron
Poisson background tuned to a ~10 Hz ground state. dt = 1 ms.

### `mouse_v1_guozhang` — the Billeh V1 core

From `~/Training-data-driven-V1/original/` (BillehColumn). **111 GLIF cell types** from the Allen
model (`network_dat.pkl`), staged in `mice_v1_guozhang_neurons.npz`: V_th ∈ [−58, −16], E_L =
V_reset ∈ [−85, −62], C_m 100–190 pF, **τ 4.5–63.7 ms**, two after-spike currents. **Configurable
size**: `load_mice_v1_glif(n_neurons=…)` sub-samples the r<400 µm core (full core **51,978**
neurons / 14.4M synapses) — random, or closest-to-centre (`connected_selection`) — as the upstream
`load_sparse.py`. See [mice_v1_guozhang.md](mice_v1_guozhang.md).

### `mouse_column_v1` — the generated VISp column

From `~/src/mice_unnamed_torch_dev`. **4,166 neurons, 726,404 signed edges, 21 cell types** (pyr /
PV / SST / VIP / rest × L23/L4/L5/L6 + L5 ET/IT; 3,049 E / 1,117 I). Per-cell-type params are the
Allen **VISp GLIF** fits (`glif_models_VISp/`, averaged per type; τ 7.4–46.8 ms), staged in
`mice_column_v1_neurons.npz`. **Size scales by `--replicate`** — block-diagonal tiling of the
column — keeping its generated connectivity, not radial subsampling. See
[mice_column_v1.md](mice_column_v1.md).
