# Runtime Graph Replication

Any square catalog graph can be expanded at runtime by block-diagonal
replication. This creates independent copies of the same graph:

```text
A_rep = block_diag(A, A, ..., A)
```

There are no inter-copy edges. This is useful for scaling sparse kernels and
RSNN benchmarks without generating new biological data.

## Supported Commands

PyTorch SpMV/SpMM:

```bash
python -m connectome_dataset.benchmarks.bench_spmv \
  --graph mice_column_v1 \
  --replicate 16 \
  --device cuda
```

JAX SpMV/SpMM:

```bash
python -m connectome_dataset.benchmarks.bench_spmv_jax \
  --graph mouse_retina_1 \
  --replicate 8
```

Available JAX SpMV providers:

- `jax_bcoo`
- `brainevent_csr`
- `brainevent_event`

PyTorch RSNN sparse mode:

```bash
python -m connectome_dataset.benchmarks.bench_rsnn \
  --mode sparse \
  --graph mice_column_v1 \
  --replicate 4 \
  --device cuda
```

JAX/brainevent RSNN sparse mode:

```bash
python -m connectome_dataset.benchmarks.bench_rsnn_brainstate \
  --mode sparse \
  --graph mice_column_v1 \
  --replicate 4
```

The RSNN JAX benchmark supports explicit backend selection:

```bash
python -m connectome_dataset.benchmarks.bench_rsnn_brainstate \
  --mode sparse \
  --graph mice_column_v1 \
  --sparse-backend brainevent
```

```bash
python -m connectome_dataset.benchmarks.bench_rsnn_brainstate \
  --mode sparse \
  --graph mice_column_v1 \
  --sparse-backend jax_bcoo
```

## API

```python
from connectome_dataset import load_catalog, find_graph, replicate_graph_entry

catalog = load_catalog()
entry = find_graph("mice_column_v1")
rep_entry, matrix = replicate_graph_entry(entry, catalog, times=16)

print(rep_entry["name"], matrix.shape, matrix.nnz)
```

## Constraints

- The source graph must be square.
- `--replicate` requires `--graph` for SpMV commands to avoid accidentally
  expanding the whole catalog.
- RSNN sparse mode defaults to the catalog `mice_column_v1` entries; use
  `--graph` when requesting a custom replication factor.
- `--sparse-backend auto` prefers `brainevent` and falls back to `jax_bcoo`
  if the `brainevent` path is unavailable.
- Practical scale is bounded by CPU RAM, GPU memory, sparse-matrix conversion
  cost, and backend index limits.

## Catalog Policy

Replicas are runtime-only. Do not add fixed replicated variants to
`graphs/index.json`; use `--replicate N` instead.
