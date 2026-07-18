# mice_column_v1

`mice_column_v1` is the repo-local name for Guozhang's mouse V1 cortical-column
network used by the sparse RSNN benchmarks.

## Summary

| Field | Value |
|-------|-------|
| Species | mouse |
| Region | V1 / VISp cortical column |
| Neurons | 4,166 |
| Directed edges | 726,404 |
| Source variant | `generated_v1_4166_726404_uuid_62e03c29` |
| Column bounds | x `[2000, 2200]`, z `[2200, 2400]` |
| Excitatory neurons | 3,049 |
| Inhibitory neurons | 1,117 |

Layer counts:

| Layer | Count |
|-------|------:|
| L23 | 1,105 |
| L4 | 811 |
| L5 | 1,105 |
| L6a | 951 |
| L6b | 194 |

## Files

Tracked metadata:

- `datasets/mice_column_v1/manifest.json`
- `datasets/mice_column_v1/network_generation.yaml`
- `datasets/mice_column_v1/rsnn_params.yaml`

DVC-managed payload:

- `data/external/mice_column_v1.dvc`
- `data/external/mice_column_v1/mice_connections_processed.parquet`
- `data/external/mice_column_v1/mice_neurons_processed.parquet`
- `data/external/mice_column_v1/generation_params.yaml`

## Loading

```python
from connectome_dataset import load_mice_column_v1

adj = load_mice_column_v1()
weighted = load_mice_column_v1(use_weights=True)
```

The default load returns binary topology. Use `use_weights=True` to load the
stored signed synaptic weights.

## Catalog Entry

The generated catalog includes only `mice_column_v1`. For larger replicated
networks, use runtime replication:

```bash
python -m connectome_dataset.benchmarks.bench_spmv \
  --graph mice_column_v1 \
  --replicate 16
```

## Provenance Note

The source documentation names `guozhang_4166_726404_uuid_bcc04c42` as the
current default. Local inspection found that directory's processed parquet files
contain 724,159 edges, while `generated_v1_4166_726404_uuid_62e03c29` contains
726,404 edges and matches the prior catalog lineage. This repo uses the
726,404-edge runtime copy for `mice_column_v1`.

## Dependency Boundary

This repo must not import code from, or read runtime data from, the original
development checkout. Required metadata is copied into `datasets/mice_column_v1/`
and large payloads are represented through DVC under `data/external/`.
