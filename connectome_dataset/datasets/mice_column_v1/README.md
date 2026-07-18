# mice_column_v1

Minimal repo-local metadata for Guozhang's mouse V1 cortical-column network.

See `docs/mice_column_v1.md` for the organized user-facing dataset document and
`docs/data_management.md` for DVC/data policy.

## Identity

- Name: `mice_column_v1`
- Species: mouse
- Region: V1 / VISp cortical column
- Neurons: 4,166
- Directed edges: 726,404
- Source variant: generated/runtime Guozhang copy, UUID `62e03c29`
- Column bounds: x `[2000, 2200]`, z `[2200, 2400]`
- Neuron class counts: 3,049 excitatory, 1,117 inhibitory
- Layers: L23 1,105; L4 811; L5 1,105; L6a 951; L6b 194

The source documentation also names
`guozhang_4166_726404_uuid_bcc04c42` as the current default. On local
inspection that directory's processed parquet files contain 724,159 edges,
while `generated_v1_4166_726404_uuid_62e03c29` contains 726,404 edges and
matches the existing catalog lineage. This repo uses the 726,404-edge runtime
copy for `mice_column_v1`.

## In-Repo Data

Tracked, small, commit-safe files live here:

- `manifest.json`: expected external files, byte sizes, hashes, and summary stats.
- `network_generation.yaml`: sanitized Guozhang network-generator parameters.
- `rsnn_params.yaml`: minimal BaseRSNN/Hydra defaults copied from the source docs.

These files are sufficient to identify the dataset and reconstruct the loader
contract without importing or reading any files from `mice_unnamed_torch_dev`.

## External Data

Large data is DVC-managed at:

```text
data/external/mice_column_v1/
```

Expected files:

- `mice_connections_processed.parquet`
- `mice_neurons_processed.parquet`
- `generation_params.yaml`

Git tracks `data/external/mice_column_v1.dvc`, not the parquet files. Restore
or update the data with:

```bash
dvc pull data/external/mice_column_v1.dvc
dvc add data/external/mice_column_v1
```

No DVC remote is configured yet. Before sharing the payload, configure a remote
and run `dvc push data/external/mice_column_v1.dvc`.

Runtime code must read only from this repo-local external-data path or from
`CONNECTOME_DATA_ROOT`, never from `mice_unnamed_torch_dev`.
