# mice_v1_guozhang

Repo-local metadata for the **Guozhang V1 GLIF model core** — the canonical
Billeh-style mouse V1 column, kept as one preprocessed neuron-to-neuron
adjacency for larger-scale sparse GPU benchmarks. It complements
[`mice_column_v1`](../mice_column_v1/README.md): same V1 family, ~12x the
neurons and ~20x the edges, so a kernel's ranking can be read as network size
grows (and further via `--replicate`).

## Identity

- Name: `mice_v1_guozhang`
- Species: mouse
- Region: V1
- Neurons: 51,978 (radial core, `r < 400` um)
- Directed edges: 14,441,124 (density 0.53%)
- Weights: signed synaptic weights, `[-25.8, 927.1]`, mean 3.07
- Orientation: rows = presynaptic, cols = postsynaptic (same as `mice_column_v1`)

## Preprocessing

Built once from the raw GLIF V1 model by `scripts/build_mice_v1_guozhang.py`:

1. select the model **core** (`r = sqrt(x^2 + z^2) < 400` um → the 51,978-neuron
   V1 column),
2. sum the four GLIF **receptor-type channels** into one signed weight per
   (presynaptic, postsynaptic) pair,
3. store as a scipy CSR `.npz`.

Raw source (machine-local, not vendored): `/data/fanqixuan/dataset/GLIF_V1_network`
(`network_dat.pkl`, `network/v1_nodes.h5`). Only the connectivity is kept — the
1.7 GB pickle and the stimulus payloads are not copied into the repo.

Rebuild:

```bash
micromamba run -n ml-py312 python scripts/build_mice_v1_guozhang.py \
    --src /data/fanqixuan/dataset/GLIF_V1_network
```

## In-Repo Data

Tracked, small, commit-safe files live here:

- `manifest.json`: expected external files, byte sizes, hashes, and summary stats.

## External Data

The preprocessed payload is DVC-managed under `data/external/mice_v1_guozhang/`:

- `mice_v1_guozhang.npz` — CSR adjacency (float32, signed weights)
- `generation_params.yaml` — build provenance

Git tracks `data/external/mice_v1_guozhang.dvc`, not the `.npz`. Restore or update:

```bash
dvc pull data/external/mice_v1_guozhang.dvc
dvc push data/external/mice_v1_guozhang.dvc   # after (re)building
```
