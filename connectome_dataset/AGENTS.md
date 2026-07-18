# Repository Rules

## Data Layout

- `datasets/` is for small, git-tracked metadata/config/provenance only.
- `data/` is for large runtime payloads restored by download scripts, symlinks,
  or DVC.
- Commit `.dvc` pointer files such as `data/external/*.dvc`, not the raw
  payload directories they reference.
- Do not add large parquet, zip, tar, Matrix Market, GraphML, HDF5, or NumPy
  payloads to `datasets/`.
- Runtime code must not depend on sibling development checkouts. Copy required
  metadata here and manage large payloads through `data/` plus DVC/downloaders.

## Package Layout

- The installable Python package is `connectome_dataset/` at the repo root.
- Do not recreate a `src/` package layout.
