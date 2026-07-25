# mice_v1_guozhang

`mice_v1_guozhang` is the repo-local name for the **core of Guozhang's GLIF mouse
V1 model** — the canonical Billeh-style V1 column, kept as one preprocessed
neuron-to-neuron adjacency for larger-scale sparse GPU benchmarks. It is the same
V1 family as [`mice_column_v1`](mice_column_v1.md), ~12× the neurons and ~20× the
edges.

## Summary

| Field | Value |
|-------|-------|
| Species | mouse |
| Region | V1 |
| Neurons | 51,978 (radial core, `r < 400` um) |
| Directed edges | 14,441,124 (density 0.53%) |
| Weights | signed synaptic weights, `[-25.8, 927.1]`, mean 3.07 |
| Orientation | rows = presynaptic, cols = postsynaptic |
| Format | scipy CSR `.npz` (float32) |

## Preprocessing

Built once from the raw GLIF V1 model by `scripts/build_mice_v1_guozhang.py`:

1. select the model **core** (`r = sqrt(x^2 + z^2) < 400` um → the 51,978-neuron
   V1 column),
2. sum the four GLIF **receptor-type channels** into one signed weight per
   (presynaptic, postsynaptic) pair,
3. store as a scipy CSR `.npz`.

Raw source (machine-local, not vendored): `/data/fanqixuan/dataset/GLIF_V1_network`
(`network_dat.pkl`, `network/v1_nodes.h5`). Only the connectivity is kept — the
1.7 GB pickle and stimulus payloads are not copied into the repo.

Rebuild:

```bash
micromamba run -n ml-py312 python scripts/build_mice_v1_guozhang.py \
    --src /data/fanqixuan/dataset/GLIF_V1_network
```

## Files

Tracked metadata:

- `datasets/mice_v1_guozhang/manifest.json`
- `datasets/mice_v1_guozhang/README.md`

DVC-managed payload:

- `data/external/mice_v1_guozhang.dvc`
- `data/external/mice_v1_guozhang/mice_v1_guozhang.npz`
- `data/external/mice_v1_guozhang/generation_params.yaml`

Restore or update the payload:

```bash
dvc pull data/external/mice_v1_guozhang.dvc
dvc push data/external/mice_v1_guozhang.dvc   # after (re)building
```

## Benchmarks

Part of the `v1` matrix set (with `mice_column_v1`):

```bash
connectome-bench --device cuda --matrix-set v1
# or the demo job:
sbatch scripts/bench_v1.sbatch
```
