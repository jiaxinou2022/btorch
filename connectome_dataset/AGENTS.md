# AGENTS.md

Single source of truth for working in this repo. `CLAUDE.md` just points here.

## Coding style

- Clean, clear, copy-paste-experimentable code. A reader should be able to lift a
  function into a REPL and run it.
- [Google Python style](https://google.github.io/styleguide/pyguide.html):
  Google-format docstrings, imports at module top, `snake_case` functions,
  explicit names over clever ones.
- Decide visibility deliberately. Prefix with `_` only for genuinely internal
  helpers that will not be reused; if a function may plausibly be called from
  another module, notebook, or test later, keep it public. Avoid a wall of
  `_xx` helpers.
- Keep it concise — short functions, minimal ceremony, comments only where the
  code cannot speak for itself.

## What this repo is

A connectome graph catalog plus a **sparse-matmul / RSNN benchmark suite**. Two
benchmark pillars over connectome and connectome-like graphs:

1. **Sparse matmul** — SpMM (`C = A·B`, sparse×dense; `N=1` is SpMV) and
   SpGEMM (`C = A·Aᵀ`, sparse×sparse), comparing SOTA algorithms across
   frameworks and GPU tiers.
2. **End-to-end RSNN** — a recurrent spiking network whose inner loop is a
   fixed-operator SpMM/SpMV over many timesteps.

The suite supports **in-tree baselines** (JAX, torch, cupy/cuSPARSE, NEST, brian2) and
**out-of-tree custom kernels** (Sputnik, FlashSparse, DTC-SpMM, …), with a
cross-language **performance history** and an **algorithm leaderboard**.

## Environment — always use micromamba

Every Python command runs through the conda env `ml-py312`:

```bash
micromamba run -n ml-py312 <command>
```

This applies to pytest, cmake, and any script invocation. Editable install:
`micromamba run -n ml-py312 pip install -e .`

## Common commands

```bash
# Catalog (machine-local absolute paths → .cache/graph_catalog.json, gitignored;
# rebuild on each machine)
connectome-catalog                       # or: python -m connectome_dataset.catalog

# Full benchmark run: shells out to pytest per framework (separate processes),
# then ingests history + rewrites the leaderboard
connectome-bench [--device cuda|cpu] [--quick] [--mode dense|sparse|all]

# Run ONE framework directly (JAX and torch MUST stay in separate processes —
# CUDA-context reasons; connectome-bench enforces this by shelling out twice)
micromamba run -n ml-py312 python -m pytest connectome_dataset/benchmarks/jax   -v -s --benchmark-autosave
micromamba run -n ml-py312 python -m pytest connectome_dataset/benchmarks/torch -v -s --benchmark-autosave

# A single test
micromamba run -n ml-py312 python -m pytest connectome_dataset/benchmarks/torch/test_spmv.py -v -s -k native_sparse

# History / leaderboard (thin unification layer over pytest-benchmark + Google Benchmark)
connectome-bench-report ingest --pytest ".benchmarks/**/*.json" --cpp "results/cpp/**/records.jsonl"
connectome-bench-report report           # → results/report.html + results/leaderboard.md

# Lint (ruff is used in this repo)
micromamba run -n ml-py312 ruff check .
```

Useful pytest CLI options (see `benchmarks/conftest.py`): `--graph <id>`,
`--replicate N`, `--batch-size`, `--timesteps`, `--device`, `--mode`,
`--precisions fp32,fp16,bf16`, `--synthetic`, `--matrix-set <name>`,
`--warmup`/`--rounds`.

### C++ harness

```bash
micromamba run -n ml-py312 cmake -B cpp/build -S cpp
micromamba run -n ml-py312 cmake --build cpp/build -j4
cpp/build/connectome_bench --matrix path.mtx --target spmv --batch 32   # or --target spgemm
# opt-in Google Benchmark micro-bench:
micromamba run -n ml-py312 cmake -B cpp/build -S cpp -DCONNECTOME_BENCH_BUILD_BENCHMARKS=ON
```

The C++ runner writes the **same unified record schema** (`records.jsonl` +
`records.csv` + `manifest.json`) so its results ingest into the same leaderboard.

## Architecture

### Package layout (hard rules)

- The installable package is `connectome_dataset/` at the repo root. **Do not**
  create a `src/` layout.
- `datasets/` — small, git-tracked metadata/config/provenance only. No large
  parquet, zip, tar, Matrix Market, GraphML, HDF5, or NumPy payloads.
- `data/` — large runtime payloads; gitignored, restored by download scripts or
  DVC. Commit `.dvc` pointer files (`data/external/*.dvc`), never the raw
  parquet/zip/tar/GraphML/HDF5/npy payloads.
- Runtime code must not depend on sibling development checkouts — copy needed
  metadata into `datasets/`, manage large payloads via `data/` + DVC/downloaders.

### Data flow

`catalog.py` scans data roots → writes `.cache/graph_catalog.json` (machine-local).
`graph_loader.py` loads any supported format (GraphML, skewed.de CSV, parquet,
conn2res, SuiteSparse Matrix Market, synthetic block-diagonal tiling) into a
scipy CSR matrix and computes `graph_properties`. Runtime replication expands any
square graph via block-diagonal tiling without new catalog entries.

### Benchmark tree

Mirrors the conceptual identity `framework / provider / target / variant`
(collapsed to one file when a level is small):

```
connectome_dataset/benchmarks/
  cases.py       # SpmvCase/SpgemmCase/RSNNCase dataclasses + loaders + stratification()
  config.py      # per-target defaults (SPMV_DEFAULTS, RSNN_DEFAULTS)
  templates.py   # base_extra_info() (unified record schema) + measure_compile_ms + run_pedantic
  providers.py   # out-of-tree SpMM provider REGISTRY (entry points + in-process register)
  conftest.py    # CLI options + shared fixtures
  run_all.py     # `connectome-bench` entry point
  jax/  torch/  cupy/  nest/  brian2/  external/  rsnn/   # provider leaves
  history/       # ingest · store · leaderboard · report · provenance
metrics.py       # flops_spmm/spgemm, tflops, real_time_factor, events_per_s
```

Key seams to understand before editing:

- **Unified record schema** — every benchmark leaf populates
  `benchmark.extra_info` through `templates.base_extra_info(...)`. This is what
  makes JAX, torch, cupy, out-of-tree kernels, and C++ all ingestible into one
  history. Add fields here, not ad hoc in a leaf.
- **`stratification(case)`** (in `cases.py`) — computes the structural axes
  (density bucket, avg degree / degree type, gini, cov, compression ratio) that
  determine which algorithm/CUDA feature wins. Records are ranked per **workload
  cell** = `(target, case_id, problem, hardware_id)`.
- **Out-of-tree providers** (`providers.py`) — third-party kernels plug in via a
  `[project.entry-points."connectome_bench.spmm"]` entry point returning
  `SpmmProvider`(s); nothing in this repo needs editing. `register_spmm()` is the
  in-process path for notebooks/tests. A broken entry point warns and is skipped,
  never breaking discovery for others. Set `self_timed=True` when a kernel
  reports its own CUDA-event kernel time (so wall-clock H2D/D2H isn't recorded).
  The taxonomy rule: **framework = the invoking runtime** — a torch extension is
  `torch.*`, a native kernel called via ctypes is `cpp.*`.
- **`history/`** is a *thin* unification layer — it does not re-time or re-store,
  only canonicalizes cross-language records, ranks, and emits static reports.
  History appends to `results/history.jsonl`.

### JAX vs torch process isolation

`connectome-bench` deliberately shells out to pytest once per framework because
JAX and torch cannot share a CUDA context in-process. Keep new framework leaves
in their own directory and let `run_all.py` invoke them as separate pytest runs.

## Docs

Practical guide: `docs/benchmarks.md`. Design rationale, literature, and the
current handoff state live in `docs/benchmark_design/` (see `HANDOFF.md`).
Data management: `docs/data_management.md`. Per-dataset notes:
`docs/{mice_column_v1,microns,suitesparse,flywire}.md`.
