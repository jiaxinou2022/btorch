# connectome-dataset

Connectome graph catalog and a sparse-matmul / RSNN benchmark suite.

## Setup

```bash
pip install -e .
```

All Python in this repo is run through the conda env:

```bash
micromamba run -n ml-py312 <command>
```

## Data

Download connectome data with the scripts in `scripts/`:

```bash
python scripts/download_connectomes.py       # NeuroData GraphML (mouse, fly, worm, cat, macaque, rat)
python scripts/download_skewed_connectomes.py  # networks.skewed.de CSV (hemibrain, fly larva, C.elegans…)
python scripts/download_conn2res.py          # conn2res Zenodo (drosophila, macaque, 6× human consensus)
python scripts/download_suitesparse.py --groups Arenas Newman DIMACS10 --max-nnz 100000 --dvc
dvc pull data/external/microns.dvc           # MICrONS mm3 HDF5 payload
```

Data rule:

- `datasets/` — small, git-tracked metadata / config / provenance only.
- `data/` — large runtime payloads; ignored, downloaded, or DVC-managed.
- `data/external/*.dvc` files may be committed; raw parquet/zip/tar/graph
  payloads under `data/` must not be.

See [docs/data_management.md](docs/data_management.md).
`mice_column_v1` is Guozhang's 4,166-neuron mouse V1 column network
([docs/mice_column_v1.md](docs/mice_column_v1.md)). SuiteSparse groups
`DIMACS10`, `Arenas`, `Newman` are supported via Matrix Market archives
([docs/suitesparse.md](docs/suitesparse.md)).

## Catalog

The catalog is machine-local (absolute paths), so it is written to `.cache/`
and gitignored — rebuild it on each machine:

```bash
connectome-catalog                      # writes .cache/graph_catalog.json
python -m connectome_dataset.catalog    # same
```

## Benchmarks

Two pillars over connectome and connectome-like graphs:

1. **Sparse matmul** — SpMM (`C = A·B`, sparse×dense; `N=1` is SpMV) and
   SpGEMM (`C = A·Aᵀ`, sparse×sparse), comparing SOTA algorithms.
2. **End-to-end RSNN** — a recurrent spiking network whose inner loop is a
   recurrent, fixed-operator SpMM/SpMV over many timesteps.

The suite supports **in-tree baselines** and **out-of-tree custom kernels**,
across GPU tiers, with a **trackable performance history** and an **algorithm
leaderboard**. See **[docs/benchmarks.md](docs/benchmarks.md)** for the practical
guide (corpora, providers, running, reading results, adding a kernel); the design
rationale and literature live in [docs/benchmark_design/](docs/benchmark_design/).

### Running

```bash
connectome-bench [--device cuda|cpu] [--quick]   # runs jax + torch (separate processes),
                                                 # then ingests history + writes the leaderboard
```

`connectome-bench` shells out to pytest twice (JAX and torch must stay in
separate processes for CUDA-context reasons), autosaves each run via
`pytest-benchmark`, then ingests the new results and rewrites the report.

Run one framework directly with pytest:

```bash
micromamba run -n ml-py312 python -m pytest connectome_dataset/benchmarks/jax   -v -s --benchmark-autosave
micromamba run -n ml-py312 python -m pytest connectome_dataset/benchmarks/torch -v -s --benchmark-autosave
```

### Layout

The benchmark tree mirrors the conceptual identity
`framework / provider / target / variant` (collapsed to a single file when a
level is small):

```
connectome_dataset/benchmarks/
  cases.py           # Spmv/Spgemm/RSNN cases + loaders + stratification()
  config.py          # shared defaults
  templates.py       # base_extra_info (unified record schema), measure_compile_ms, run_pedantic
  conftest.py        # CLI options, shared fixtures
  run_all.py         # `connectome-bench` entry point
  jax/               # jax_native + brainevent providers (spmv, brainevent/rsnn)
  torch/             # native_sparse / pytorch_sparse providers (spmv), btorch/rsnn
  history/           # ingest · store · leaderboard · report · provenance
```

`../metrics.py` holds the FLOP/throughput helpers (`flops_spmm/spgemm`,
`tflops`, `real_time_factor`, `events_per_s`).

### History & leaderboard

`history/` is a thin unification layer over the native tools
(`pytest-benchmark` for Python, Google Benchmark for C++): it does not rebuild
timing or storage, only cross-language canonicalization, ranking, and static
reports. Records from every producer are normalized into one schema and ranked
per **workload cell** = `(target, case_id, problem, hardware_id)`.

```bash
connectome-bench-report ingest --pytest ".benchmarks/**/*.json" \
    --gbench run.json --cpp "results/cpp/**/records.jsonl"
connectome-bench-report report        # -> results/report.html + results/leaderboard.md
```

History is appended to `results/history.jsonl`.

## C++ harness

The C++ core lives under `cpp/` and installs `connectome_bench::core` through
CMake for both in-tree use and external (`find_package`) integration. It follows
a Ginkgo-style design: a `Reference` executor is the numerical oracle every
accelerated kernel is validated against, with interchangeable implementations
behind one interface and a registry.

```bash
micromamba run -n ml-py312 cmake -B cpp/build -S cpp
micromamba run -n ml-py312 cmake --build cpp/build -j4
cpp/build/connectome_bench --matrix path.mtx --target spmv --batch 32   # or --target spgemm

# opt-in Google Benchmark micro-bench:
micromamba run -n ml-py312 cmake -B cpp/build -S cpp -DCONNECTOME_BENCH_BUILD_BENCHMARKS=ON
```

The C++ runner writes records in the same unified schema (`records.jsonl` +
`records.csv` + `manifest.json`), so its results ingest into the same
leaderboard as the Python runs.

## Runtime replication

Any square catalog graph can be expanded at runtime by block-diagonal
replication — independent copies of the same graph without new catalog entries.
See [docs/runtime_replication.md](docs/runtime_replication.md).
