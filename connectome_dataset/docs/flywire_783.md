# flywire_783 — whole-brain *Drosophila* connectome + LIF model

`flywire_783` is the entire adult *Drosophila melanogaster* brain (FlyWire v783):
**138,639 neurons, 15,091,983 signed synaptic connections** — the largest graph in
the corpus. It is the wiring of the Shiu et al. (2024) leaky-integrate-and-fire (LIF)
computational brain model, and this repo integrates both the connectome (as a
benchmark graph) and the model (reimplemented in btorch, no brian2).

- Paper: Shiu et al. (2024) *A leaky integrate-and-fire computational model based on
  the connectome of the entire adult Drosophila brain reveals insights into
  sensorimotor processing.* Nature 634:210–219. doi:10.1038/s41586-024-07763-9
- Upstream: <https://github.com/philshiu/Drosophila_brain_model>

## The connectome (benchmark graph)

Loaded straight from the model's own tables, in their original format — no `.npz`
repackaging, so the numbers match the published model:

| File | Contents |
|------|----------|
| `Completeness_783.csv` | neuron list, indexed by FlyWire ID (row/col index `i` ↔ `flyid`) |
| `Connectivity_783.parquet` | synapse table; edge weight = `Excitatory x Connectivity` (signed synapse count, `<0` inhibitory) |

`connectome_dataset.graph_loader.load_flywire_783()` builds the N×N signed CSR
(rows = presynaptic, cols = postsynaptic). Staging (repo does not depend on any
external checkout):

```bash
mkdir -p data/external/flywire_783
cp Completeness_783.csv Connectivity_783.parquet data/external/flywire_783/
micromamba run -n ml-py312 connectome-catalog        # registers flywire_783
```

Structure: density 0.079%, ~40% inhibitory edges, in-degree Gini 0.53 (heavy-tailed
hubs), max in-degree 10,356. As a benchmark graph it is the large-scale sparse
stress case — an order of magnitude past `fly_hemibrain` (21.7k) and `mice_v1_guozhang`
(52k).

## The model (btorch)

`connectome_dataset/benchmarks/torch/btorch/flybrain.py` reimplements the Shiu LIF
dynamics on btorch primitives (`LIF` + `SparseConn`), with constants lifted verbatim
from upstream `model.py`:

| symbol | value | meaning |
|--------|-------|---------|
| `v_0 = v_rst` | −52 mV | resting = reset potential |
| `v_th` | −45 mV | spike threshold |
| `t_mbr` | 20 ms | membrane time constant |
| `tau` | 5 ms | synaptic time constant |
| `t_rfc` | 2.2 ms | refractory period |
| `t_dly` | 1.8 ms | synaptic delay |
| `w_syn` | 0.275 mV | weight per unit of signed connectivity |
| `r_poi` | 150 Hz | Poisson activation rate |

Because `v_0 == v_rst`, the membrane leaks toward its own reset, so this is exactly
btorch `LIF(v_reset=−52, tau=c_m=t_mbr)` driven by current `I = g`; the recurrent
`g += spikes @ W·w_syn` is `SparseConn`, and the Shiu reset `g = 0` on spike is applied
each step. **Activation** models optogenetics: upstream's Poisson `v`-kick
(`w_syn·f_poi = 68.75 mV`, refractory 0) forces a spike per event, which we reproduce
dt-robustly by emitting a Poisson spike train at `r_poi` from the activated neurons.
**Silencing** zeros every synapse to and from a neuron (its adjacency row and column).

```python
from connectome_dataset.graph_loader import load_flywire_783
from connectome_dataset.benchmarks.torch.btorch.flybrain import FlyBrainModel

W, flyids = load_flywire_783(return_ids=True)
model = FlyBrainModel(W, flyids, device="cuda")
rates = model.run_experiment(activate=neu_sugar, t_run=1000.0, n_run=30)  # Hz per neuron
```

### Validation

Cross-checked against the **brian2** reference — a native port of the upstream `model.py`
onto the same connectome, shipped as its own framework leaf
(`benchmarks/brian2/native/flybrain.py`, recorded as `brian2.native.rsnn.flybrain`, the same
leaderboard cell as the btorch model). Per-neuron firing-rate correlation **r ≈ 0.999** over
the responsive population; on the full brain both give ~0.2–0.3% active ratio and ~0.1 Hz mean
under sugar drive. btorch uses forward-Euler membrane integration where brian2 uses
exact-linear, a small (~10%) systematic amplitude offset on downstream rates; structure and
sign/delay/reset dynamics match. brian2 is optional — the leaf skips cleanly if it is absent.

### Drive types

The model's inputs are sensory neuron groups activated as optogenetic Poisson drive
(`datasets/flywire_783/drive_types.json`, from the Shiu experiments): gustatory —
`sugar`, `sugar_left`, `bitter`, `ir94e`, `water` — and auditory/mechanosensory
Johnston's-Organ — `JON_CE`, `JON_F`, `JON_D_m` (230 v783 input neurons total). Load one,
several, or all via `graph_loader.all_flywire_drive_ids()`; the benchmark drives **all
types simultaneously**.

### Tonic background

The upstream model is stimulus-only, so without drive the brain is silent and even all drives
give only ~0.4 Hz. `FlyBrainModel` adds an optional **tonic background** (`bg_rate` /
`bg_weight`, on by default) — an independent per-neuron Poisson drive into the synaptic
conductance `g` (not into `v`), tuned to the diffusion regime — that holds the network at a
low asynchronous-irregular baseline. **The default is tonic background + all drives.**

| condition | population mean | active ratio |
|---|---|---|
| no background, sugar only | 0.12 Hz | 0.3% |
| no background, all drives | 0.41 Hz | 0.9% |
| **tonic background only** | ~7.7 Hz | ~30% |
| **tonic background + all drives** | ~9.0 Hz | ~29% |

**Avoiding over/undershoot without clamping `v`:** two safeguards keep `v` bounded — `hard_reset`
(the upstream `v = v_rst`), and masking synaptic input to refractory neurons (btorch integrates
`v` every step, but the Shiu equations are `(unless refractory)`; without this a strong recurrent
hit accumulates on a frozen neuron and overshoots to +80 mV). With both, `v` never exceeds the
threshold (no overshoot). Strongly-inhibited neurons still hyperpolarise deeply — that is faithful
model behaviour, which the brian2 reference (native `(unless refractory)`) reproduces, not blow-up.
The background level is fully configurable; `bg_rate=0` recovers the stimulus-only model.

## Benchmark

`test_flybrain.py` records the full simulation on the RSNN track as
`torch.btorch.rsnn.flybrain`, alongside the `native`/`vdha`/`mh_spgemm` spike-delivery
variants and the `coba`/`cuba`/`connectome` beNNch models.
