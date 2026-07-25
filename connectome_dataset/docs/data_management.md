# Data Management

This repo separates small metadata from large graph payloads.

## Core Rule

`datasets/` and `data/` are intentionally different:

- `datasets/`: small, git-tracked dataset identity, metadata, parameters,
  manifests, and provenance.
- `data/`: large runtime payloads restored by download scripts, symlinks, or
  DVC. Raw payload files under `data/` are not commit-safe.
- `data/external/*.dvc`: small DVC pointer files that may be committed.

Do not put large parquet, zip, tar, Matrix Market, GraphML, HDF5, or NumPy
payloads in `datasets/`. Do not commit raw payloads from `data/`.

## Directory Layout

| Path | Git policy | Purpose |
|------|------------|---------|
| `graphs/index.json` | tracked | Catalog generated from available data. |
| `datasets/` | tracked | Small dataset manifests, parameters, and provenance. |
| `data/` | ignored by default | Downloaded or DVC-materialized graph payloads. |
| `data/external/*.dvc` | tracked | DVC pointer files for large external datasets. |
| `.dvc/` | tracked config, ignored cache | DVC repo metadata and local object cache. |

Large binary/parquet/zip payloads should not be committed directly to git.

## DVC Workflow

`mice_column_v1` is currently tracked with DVC:

```bash
dvc status
dvc pull data/external/mice_column_v1.dvc
dvc pull data/external/mice_v1_guozhang.dvc
dvc pull data/external/suitesparse.dvc
dvc pull data/external/microns.dvc
```

`mice_v1_guozhang` is preprocessed from a machine-local raw model rather than
downloaded; if the payload is missing and no DVC remote has it, regenerate it
with `python scripts/build_mice_v1_guozhang.py`.

To update the materialized payload after changing files under
`data/external/mice_column_v1/`:

```bash
dvc add data/external/mice_column_v1
dvc status
```

Commit the resulting `.dvc` pointer, not the payload directory.

## DVC Remote

No DVC remote is configured in this repo yet. Before sharing data across
machines, add a remote:

```bash
dvc remote add -d <name> <url>
dvc push data/external/mice_column_v1.dvc
dvc push data/external/suitesparse.dvc
dvc push data/external/microns.dvc
```

After that, a fresh checkout can restore the payload with:

```bash
dvc pull data/external/mice_column_v1.dvc
dvc pull data/external/suitesparse.dvc
dvc pull data/external/microns.dvc
```

## Environment Override

By default, loaders read from this repo's `data/` directory. To use a different
data root:

```bash
CONNECTOME_DATA_ROOT=/path/to/connectome_data python -m connectome_dataset.catalog
```

The alternate root should preserve the same internal layout, for example:

```text
$CONNECTOME_DATA_ROOT/external/mice_column_v1/
$CONNECTOME_DATA_ROOT/conn2res/
$CONNECTOME_DATA_ROOT/graphml/
```

## Rule

Runtime code must not depend on sibling development checkouts. If a dataset is
needed here, represent it with tracked metadata in `datasets/` and either DVC
or documented download scripts for the large payload.
