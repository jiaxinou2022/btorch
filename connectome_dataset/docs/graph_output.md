# Loader Output Format

Both `load_microns()` and `load_flywire()` return the same 4-tuple:

```python
adj, types, pos, node_map = load_microns(path)
```

## Return Values

| Variable | Type | Shape | Meaning |
|----------|------|-------|---------|
| `adj` | `scipy.sparse.coo_matrix` | `(n_nodes, n_nodes)` | Directed weighted adjacency matrix |
| `types` | `np.ndarray` | `(n_nodes,)` | Binary cell type: `0=Excitatory`, `1=Inhibitory` |
| `pos` | `np.ndarray` | `(n_nodes, 3)` | Normalized 3D positions (x, y, z) |
| `node_map` | `dict` | — | Mapping: original `pt_root_id` → final `0..n-1` index |

## Adjacency Matrix (`adj`)

- **Format**: `scipy.sparse.coo_matrix` (row, col, data)
- **Directed**: yes (source → target)
- **Weighted**: yes, by `synapse_volume` or `syn_count`
- **Self-loops**: typically none (depends on source data)
- **LCC extracted**: only nodes in the largest connected component are retained (if `extract_lcc=True`)

### Common post-processing: binarization

Many downstream scripts in this repo force binarize weights for spectral stability:

```python
import numpy as np
adj.data = np.ones_like(adj.data)   # discard weights, keep topology
```

## Cell Types (`types`)

| Value | Meaning | Typical Fraction (MICrONS) |
|-------|---------|---------------------------|
| `0` | Excitatory (pyramidal neurons) | ~80–85% |
| `1` | Inhibitory (interneurons) | ~15–20% |

## Positions (`pos`)

- Normalized by subtracting the mean and dividing by `max(|pos|)`.
- Range is approximately `[-1, 1]` in all axes.
- Original units were **nanometers**; normalization is unitless.

## Node Map (`node_map`)

Use this to map biological IDs back to matrix rows/cols:

```python
# Get matrix index for a specific neuron
idx = node_map[123456789]

# Get all IDs in matrix order
ids_in_order = [uid for uid, i in sorted(node_map.items(), key=lambda kv: kv[1])]
```

## Size Reference

| Dataset | ~Nodes | ~Edges (nnz) |
|---------|--------|---------------|
| MICrONS mm3 | 59,540 | ~3.1 million |
| FlyWire | ~120,000+ | varies by threshold |
