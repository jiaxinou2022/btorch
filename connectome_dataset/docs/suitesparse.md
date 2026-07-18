# SuiteSparse Graphs

This repo can download and benchmark selected graph matrices from the
SuiteSparse Matrix Collection.

Requested groups:

- `DIMACS10`: <https://sparse.tamu.edu/DIMACS10>
- `Arenas`: <https://sparse.tamu.edu/Arenas>
- `Newman`: <https://sparse.tamu.edu/Newman>

## Download

The downloader scrapes each group page and downloads Matrix Market archives to:

```text
data/external/suitesparse/
```

Default usage:

```bash
python download_suitesparse.py --groups Arenas Newman DIMACS10 --max-nnz 100000 --dvc
```

The `--max-nnz` guard is intentional. Some SuiteSparse/DIMACS10 matrices are
large. Use `--max-nnz 0` only when you intentionally want all scraped matrices:

```bash
python download_suitesparse.py --groups DIMACS10 --max-nnz 0 --dvc
```

Useful options:

| Option | Meaning |
|--------|---------|
| `--groups Arenas Newman DIMACS10` | Groups to scrape. |
| `--max-nnz 100000` | Skip matrices above this nonzero count. |
| `--limit N` | Limit downloads per group after filtering. |
| `--dry-run` | Show selected matrices without downloading. |
| `--dvc` | Run `dvc add data/external/suitesparse` after download. |

## DVC

The default downloaded subset is DVC-managed by:

```text
data/external/suitesparse.dvc
```

Restore or share it with:

```bash
dvc pull data/external/suitesparse.dvc
dvc push data/external/suitesparse.dvc
```

## Catalog

Run:

```bash
python -m connectome_dataset.catalog
```

Catalog entries are named:

```text
suitesparse_<group>_<matrix>
```

Examples:

- `suitesparse_arenas_jazz`
- `suitesparse_newman_karate`
- `suitesparse_dimacs10_delaunay_n10`

## Benchmark

SuiteSparse entries work with the same benchmark commands as other graphs:

```bash
python -m connectome_dataset.benchmarks.bench_spmv \
  --graph suitesparse_newman_karate \
  --device cpu

python -m connectome_dataset.benchmarks.bench_spmv \
  --graph suitesparse_dimacs10_delaunay_n10 \
  --replicate 8 \
  --device cuda
```
