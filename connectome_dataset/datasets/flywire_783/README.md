# flywire_783

Repo-local metadata for the **whole-brain adult *Drosophila* connectome** (FlyWire
v783) that underlies the Shiu et al. leaky-integrate-and-fire (LIF) brain model.
It is the largest connectome in the corpus — 138,639 neurons, 15.1M signed
synaptic connections — and the recurrent operator for the btorch `FlyBrain` model
([`torch/btorch/flybrain.py`](../../connectome_dataset/benchmarks/torch/btorch/flybrain.py)).

## Identity

- Name: `flywire_783`
- Species: fly (*Drosophila melanogaster*)
- Region: whole brain
- Neurons: 138,639
- Directed edges: 15,091,983 (density 0.079%)
- Weights: signed synaptic weight `Excitatory x Connectivity` (synapse count times
  the presynaptic sign), range `[-2405, 1897]`, ~40% inhibitory
- Orientation: rows = presynaptic, cols = postsynaptic

## Source data (original format, not converted)

Loaded directly from the Shiu model's own two preprocessed tables — no `.npz`
repackaging — so the numbers match the published model exactly:

| File | What it is |
|------|------------|
| `Completeness_783.csv` | the neuron list, indexed by FlyWire ID (row/col index `i` ↔ `flyid`) |
| `Connectivity_783.parquet` | the synapse table (`Presynaptic_Index`, `Postsynaptic_Index`, `Excitatory x Connectivity`, …) |

These live under `data/external/flywire_783/` (DVC-managed, not committed). They are
the FlyWire v783 public release as preprocessed by the model authors
(`philshiu/Drosophila_brain_model`); the repo does not depend on that checkout —
stage the two files into place:

```bash
mkdir -p data/external/flywire_783
cp /path/to/Drosophila_brain_model/Completeness_783.csv    data/external/flywire_783/
cp /path/to/Drosophila_brain_model/Connectivity_783.parquet data/external/flywire_783/
micromamba run -n ml-py312 connectome-catalog     # register the graph
```

## Model

The connectome is only the wiring. The dynamical model — LIF neurons, exponential
synapses, Poisson optogenetic activation, and neuron silencing — is reimplemented
in **btorch** (no brian2 runtime dependency) at
[`torch/btorch/flybrain.py`](../../connectome_dataset/benchmarks/torch/btorch/flybrain.py),
with parameters lifted verbatim from the paper. See [docs/flywire_783.md](../../docs/flywire_783.md).

## Reference

Shiu et al. (2024) *A leaky integrate-and-fire computational model based on the
connectome of the entire adult Drosophila brain reveals insights into sensorimotor
processing.* Nature 634:210–219. doi:10.1038/s41586-024-07763-9
