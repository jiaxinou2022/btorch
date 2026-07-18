# MICrONS mm3 Connectome Dataset

## File Format
- **Path**: `data/external/microns/microns_mm3_connectome.h5`
- **Format**: HDF5 with two internal datasets
- **Dependency**: `conntility` (custom wrapper for reading)
- **DVC pointer**: `data/external/microns.dvc`

Restore the payload with:

```bash
dvc pull data/external/microns.dvc
```

## Internal Datasets

| Dataset Name | Content | Object Type |
|-------------|---------|-------------|
| `full` | Complete connectivity matrix (all synapses) | `conntility.ConnectivityMatrix` |
| `condensed` | Per-neuron attributes + deduplicated connections | `conntility.ConnectivityMatrix` |

## Loading Pattern

```python
import conntility
M = conntility.ConnectivityMatrix.from_h5(path, "full")      # connectivity
C = conntility.ConnectivityMatrix.from_h5(path, "condensed") # neuron properties
```

## Vertex / Node Fields (`C.vertices`)

These fields describe individual neurons (segments).

| Field | Type | Meaning |
|-------|------|---------|
| `pt_root_id` | int64 | Unique segment identifier (Proofread root ID) |
| `cell_type` | str | Biological cell type label. Examples: `23P`, `4P`, `5P_IT`, `5P_PT`, `5P_NP`, `6IT`, `6CT`, `BC`, `MC`, `NGC`, `BPC` |
| `x_nm`, `y_nm`, `z_nm` | float | Centroid position in nanometers |
| `indegree` | int | Number of incoming synaptic connections |
| `outdegree` | int | Number of outgoing synaptic connections |

### Cell Type Mapping (E/I Binarization)

The loader maps arbitrary labels to binary excitatory/inhibitory types:

| Type | Labels | Code |
|------|--------|------|
| Excitatory (0) | `23P`, `4P`, `5P_IT`, `5P_PT`, `5P_NP`, `6IT`, `6CT`, any pyramidal | `0` |
| Inhibitory (1) | `BC` (Basket), `MC` (Martinotti), `NGC` (Neurogliaform), `BPC` (Bipolar), any label containing "inh" | `1` |

## Edge / Connectivity Fields (`M.matrix`)

| Field | Type | Meaning |
|-------|------|---------|
| `row` / `col` (sparse indices) | int | Indices into `root_ids` array |
| `data` | float | Synaptic volume (proxy for synaptic strength/weight) |
| `synapse_volume` | float | Same as `data`, exposed after DataFrame conversion |

The sparse matrix uses `root_ids` to map matrix indices to actual `pt_root_id` values:
```python
root_ids = C.vertices['pt_root_id'].values
source_id = root_ids[adj_sparse.row]
target_id = root_ids[adj_sparse.col]
```

## Notes

- The `condensed` dataset usually contains ~59K neurons.
- The `full` matrix includes multi-synapse connections between the same pair (which `sum_duplicates()` can collapse).
- Positions are in **nanometers** and span the ~100 micron EM volume.
- LCC extraction typically drops ~10–20 isolated neurons from ~59,558 → ~59,540.
