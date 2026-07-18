# FlyWire Connectome Dataset

## File Format
- **Format**: Preprocessed CSVs (derived from FlyWire v783 release)
- **Source**: Original raw files in `/data/fanqixuan/dataset/flywire/v783/` and `/data/fanqixuan/dataset/flywire_large/`

## Preprocessed Files (used by loaders)

| File | Expected Columns | Purpose |
|------|------------------|---------|
| `flywire_edges.csv` | `pre_root_id`, `post_root_id`, `syn_count` (or `source`, `target`) | Synaptic connections between segments |
| `flywire_nodes.csv` | `root_id`, `super_class` / `cell_type`, `x`, `y`, `z` (or `rep_coord_nm`) | Neuron metadata and positions |

## Raw Source Files (FlyWire v783)

These are used in `check_flywire.ipynb` for direct loading:

| File | Description |
|------|-------------|
| `neurons.csv.gz` | Base neuron table with segment IDs |
| `classification.csv.gz` | Cell type classification (`super_class`, `cell_class`, etc.) |
| `coordinates_backbone.csv.gz` | Backbone coordinates |
| `connections_princeton_no_threshold.csv.gz` | Full synapse-level connection table |
| `fafb_v783_princeton_synapse_table.csv.gz` | Large synapse table with volumes |

## Field Reference

### Edge Fields (`flywire_edges.csv`)

| Field | Type | Meaning |
|-------|------|---------|
| `pre_root_id` | int64 / str | Presynaptic segment ID (auto-renamed to `source`) |
| `post_root_id` | int64 / str | Postsynaptic segment ID (auto-renamed to `target`) |
| `syn_count` | int | Number of synapses in this connection |
| `synapse_volume` | float | Aggregate synaptic volume (optional, used as weight) |

### Node Fields (`flywire_nodes.csv`)

| Field | Type | Meaning |
|-------|------|---------|
| `root_id` | int64 | Unique segment identifier (auto-renamed to `pt_root_id`) |
| `super_class` | str | High-level classification (auto-copied to `cell_type`). Examples: `central`, `sensory`, `motor`, `unknown` |
| `cell_class` | str | Finer classification (when available) |
| `nt_type` | str | Neurotransmitter type: `ACH`, `GABA`, `GLUT`, `DA`, `SER`, `OCT` |
| `x`, `y`, `z` | float | Position in nanometers (or `rep_coord_nm` as string) |
| `rep_coord_nm` | str | Representative coordinate as `[x, y, z]` string |

## Column Renaming (handled automatically)

```
pre_root_id   -> source
post_root_id  -> target
root_id       -> pt_root_id
super_class   -> cell_type
```

## Notes

- FlyWire uses **nanometer coordinates** in the FAFB (Full Adult Fly Brain) template space.
- The `super_class` field is coarse-grained compared to MICrONS cell types.
- The loader does **not** have hardcoded E/I mappings for FlyWire classes; it relies on generic string matching (any label containing "inh" → inhibitory).
- For E/I classification from neurotransmitter, use `nt_type`: `GABA` is inhibitory, `GLUT`/`ACH` are excitatory.
