# Sparse-matmul & RSNN benchmark suite

A benchmark for **sparse matrix multiplication** and **recurrent spiking networks (RSNN)**
on connectome and connectome-like graphs, comparing SOTA algorithms across CPU/GPU, with a
trackable performance history and a static leaderboard.

- **Design rationale & literature:** [`docs/benchmark_design/`](benchmark_design/) (start with
  `README.md`, then `HANDOFF.md`).
- **This file** is the practical guide: what's in the suite, how to run it, how to read the
  results, and how to add a new kernel.

---

## 1. What it measures

Two pillars:

1. **Sparse matmul** — `C = A·B` (sparse × dense; `N=1` is SpMV) and, in progress, SpGEMM
   `C = A·Aᵀ`. `A` is a connectome adjacency (weighted, scattered, heavy-tailed degree).
2. **End-to-end RSNN** — a recurrent spiking network whose inner loop is a fixed-operator
   SpMM/SpMV over many timesteps.

The identity of every measured run is the tuple **`(framework, provider, target, variant)`**:

| axis | meaning | examples |
|------|---------|----------|
| `framework` | the runtime that **invokes** the kernel | `torch`, `jax`, `cupy`, `nest`, `brian2`, `cpp`, `scipy` |
| `provider` | the library / algorithm being benchmarked | `cusparse`, `sputnik`, `flashsparse`, `dtc_spmm`, `native_sparse` |
| `target` | the operation | `spmm` (SpMV is `spmm` with `N=1`), `rsnn` |
| `variant` | a within-provider kernel/config axis | `csr`, `coo`, `bcoo`, `event` |

**Rule:** `framework` is the *calling runtime*, not the kernel's language. FlashSparse and
DTC-SpMM are torch CUDA extensions, so they are `torch.flashsparse` / `torch.dtc_spmm`; the
out-of-tree Sputnik is native CUDA reached via ctypes, so it is `cpp.sputnik` (same as its
in-tree form). cuSPARSE through CuPy is `cupy.cusparse`; through the C++ runner it is
`cpp.cusparse`.

---

## 2. Corpora (workloads)

Selected with `--matrix-set` / `--graph` / `--synthetic` (precedence:
`--synthetic` > `--matrix-set` > `--graph`).

| selector | what | why |
|----------|------|-----|
| `--graph mice_column_v1` | a single catalog graph (default) | quick smoke / focused run |
| `--matrix-set connectome` | curated real connectomes (8 graphs) | spans size (n 213→21.7k), density (0.15%→48%), degree-skew (Gini .24→.87) |
| `--matrix-set v1` | Guozhang mouse-V1 family (`mice_column_v1`, `mice_v1_guozhang`) | same V1 wiring across two orders of scale (n 4.2k→52k); scale further with `--replicate` |
| `--matrix-set suitesparse` | Newman/DIMACS10/Arenas graph collections | literature baseline: heavy-tailed social ↔ near-uniform meshes |
| `--matrix-set snap` | SNAP power-law graphs (social/web/citation/…) | **off-distribution but connectome-like** heavy-tailed graphs, 29k→2.3M nnz |
| `--matrix-set 'suitesparse_newman_*'` | prefix glob over catalog names | ad-hoc subsets |
| `--synthetic` | controlled sweep (banded/ER/power-law/RMAT) | isolates *one* structural axis at a time under control |

The **synthetic sweep** and the **real corpora** are complementary: the synthetic generators
([`synthetic.py`](../connectome_dataset/benchmarks/synthetic.py)) hold `n` and `avg_degree`
fixed and vary only structure (verified degree-CoV spread: banded 0.02 → ER 0.25 → power-law
~2.0 → RMAT ~3.0), while the corpora ([`corpora.py`](../connectome_dataset/benchmarks/corpora.py))
sample the real distribution.

Data policy: the catalog is machine-local (`.cache/graph_catalog.json`, rebuilt with
`connectome-catalog`); large matrix payloads live under `data/` and are DVC-managed; only
`.dvc` pointers are committed. See [`data_management.md`](data_management.md) and
[`suitesparse.md`](suitesparse.md).

---

## 3. Providers (algorithms)

All SpMM providers compute the **weighted** `A·B` and are validated against an fp32 SciPy
oracle before ranking. In-tree = baselines/reference we maintain; out-of-tree = third-party
kernels reached through the entry-point registry (§6).

| identity | class | precision | notes |
|----------|-------|-----------|-------|
| `cpp.reference.eigen_csr` | reference oracle (C++/Eigen) | fp32 | correctness ground truth |
| `cpp.cusparse` | cuSPARSE (C++) | fp32/fp16/bf16 | real half compute types |
| `cupy.cusparse` | cuSPARSE (CuPy) | fp32/fp16 | designated `×baseline` |
| `torch.native_sparse` | torch CSR/COO | fp32/fp16/bf16 | framework baseline |
| `torch.pytorch_sparse` | torch-sparse | fp32 | framework baseline |
| `jax.jax_native` / `jax.brainevent` | JAX BCOO / brainevent | fp32 | incl. event-driven |
| `cpp.sputnik` | **Sputnik** (CUDA-core SOTA) | fp32 | load-balanced 1-D tiling |
| `torch.flashsparse` | **FlashSparse** (tensor-core SOTA) | fp16 | N must be a multiple of 8 |
| `torch.dtc_spmm` | **DTC-SpMM** (tensor-core SOTA) | fp32 | N must be a multiple of 16 |

The three SOTA kernels are vendored from external repos (§7) and wired **both in-tree and
out-of-tree** as cross-checks of the integration paths.

Two further targets have their own providers (same registry, cases, oracle, record schema):

**SpGEMM** (`C = A·Aᵀ`, sparse×sparse), validated against an fp64 SciPy oracle:

| identity | class | precision | notes |
|----------|-------|-----------|-------|
| `cupy.cusparse` (spgemm) | cuSPARSE SpGEMM (CuPy) | fp32/fp16 | designated `×baseline` |
| `cuda.mh_spgemm` | **MH-SpGEMM** (masking + hashing SOTA) | fp32 | out-of-tree; `A·Aᵀ` via `B = Aᵀ`; fp16 disabled (vendored fp64-centric kernel faults on some graphs) |

**SpMSpV** (`y = A·x`, sparse matrix × **sparse vector** — event-driven spike delivery),
swept over vector sparsities `0.01/0.05/0.10/0.20`, validated against a SciPy oracle:

| identity | class | precision | notes |
|----------|-------|-----------|-------|
| `cupy.cusparse` (spmspv) | cuSPARSE dense-vector SpMV | fp32/fp16 | sparsity-insensitive baseline |
| `cuda.vdha` | **VDHA** (vector-driven hash aggregation SOTA) | fp32/fp16 | out-of-tree; A in CSC; fp16 halves value traffic, accumulates in fp32 |

Both also plug into the **btorch connectome RSNN** as the recurrent spike-delivery operator
(`spikes @ W`, the beNNch *deliver* phase) — recorded as the `variant` axis of
`torch.btorch.rsnn`: `native` (torch sparse SpMV), `vdha` (SpMSpV, one spike vector),
`mh_spgemm` (SpGEMM, batch of sparse spike vectors). See
`torch/btorch/test_rsnn_deliver.py`.

See [`benchmark_design/spgemm_spmspv.md`](benchmark_design/spgemm_spmspv.md) for both
algorithms, the SpMSpV target, the RSNN delivery track, and measurement caveats.

### Precision knob
`--precisions fp32,fp16,bf16` sweeps precision (a `knobs` axis, so a kernel only ever ranks
against others at the **same** precision; fp32 is the implicit default). Reduced-precision
results are validated against the fp32 oracle within a format tolerance; a fast-but-wrong
kernel is recorded `status="incorrect"` and excluded. Finding so far: on scattered connectome
sparsity, fp16/bf16 do **not** help SpMM — it is memory/index-bound, not compute-bound.

---

## 4. Running

Always use the conda env: `micromamba run -n ml-py312 <cmd>`.

```bash
# build the machine-local catalog
connectome-catalog

# full Python suite (jax + torch + cupy + external, separate processes) + auto leaderboard
connectome-bench --device cuda                       # or --quick, --device cpu
connectome-bench --device cuda --matrix-set connectome
connectome-bench --device cuda --synthetic
connectome-bench --device cuda --matrix-set snap --precisions fp32

# report from the accumulated history
connectome-bench-report report --out results/report.html --markdown results/leaderboard.md
```

Results append to `results/history.jsonl`; the report renders `results/report.html`
(self-contained) + `results/leaderboard.md`. `results/` is machine-local (gitignored).

### GPU / Slurm
Batch scripts live in `scripts/` (partition `debug`, one GPU):

| script | runs |
|--------|------|
| `bench.sbatch` | catalog graph · synthetic sweep · connectome corpus |
| `bench_precision.sbatch` | connectome corpus × fp32/fp16/bf16 |
| `bench_sputnik.sbatch` | build + run Sputnik vs cuSPARSE |
| `bench_flashsparse.sbatch` | build + run FlashSparse (in-tree + out-of-tree) |
| `bench_dtc.sbatch` | build + run DTC-SpMM (in-tree + out-of-tree) |
| `bench_external.sbatch` | out-of-tree registry validation |
| `bench_mh_vdha.sbatch` | build + validate + run MH-SpGEMM vs cuSPARSE (SpGEMM) and VDHA vs cuSPARSE SpMV (SpMSpV) |

```bash
sbatch scripts/bench.sbatch          # queues; results land in results/slurm/<job>.out
```

### C++ harness
```bash
cmake -B cpp/build -S cpp && cmake --build cpp/build -j8
cpp/build/connectome_bench --matrix path.mtx --target spmv --batch 32
# optional CUDA baselines:
cmake -B cpp/build -S cpp -DCONNECTOME_BENCH_BUILD_CUDA=ON       # cuSPARSE (fp16/bf16)
cmake -B cpp/build -S cpp -DCONNECTOME_BENCH_BUILD_SPUTNIK=ON    # Sputnik (needs external/)
```

---

## 5. Reading the leaderboard

A **workload cell** is `(target, case_id, problem, hardware_id)` — only implementations in the
same cell are comparable. Per cell, the primary metric is TFLOP/s (SpMM) or real-time factor
(RSNN); rankings show speedup vs the fastest (`×best`) and vs the designated baseline provider
`cusparse` (`×baseline`). The aggregate is a geomean across cells, kept separate for `×best`
(always defined) and `×baseline` (only cells containing cuSPARSE). `status != ok` records are
excluded. Self-timed kernels (FlashSparse, DTC-SpMM) report their own kernel-only CUDA-event
time (`self_timed_ms`), which ingest prefers over transfer-polluted wall-clock.

---

## 6. Adding a kernel (out-of-tree)

A third-party kernel plugs in with **no change to this repo**: declare an entry point in your
package and return an `SpmmProvider`.

```toml
# your package's pyproject.toml
[project.entry-points."connectome_bench.spmm"]
mykernel = "my_pkg:get_providers"
```

```python
from connectome_dataset.benchmarks.providers import SpmmProvider

def _make_fn(matrix, *, variant, device, batch_size, precision, x_np):
    # build state once; return a callable that does the multiply and returns the result
    ...
    return lambda: my_kernel(...)          # or (result, kernel_ms) if self_timed

def get_providers():
    return [SpmmProvider(framework="torch", provider="mykernel", make_fn=_make_fn,
                         variants=["csr"], self_timed=False,
                         supports=lambda variant, precision: precision == "fp32")]
```

`pip install` it, then `pytest connectome_dataset/benchmarks/external/ --device cuda` (or
`connectome-bench`) discovers and benchmarks it through the same cases, oracle validation, and
schema as the in-tree baselines. See
[`examples/connectome_bench_scipy`](../examples/connectome_bench_scipy) for a minimal template
and `connectome_bench_{sputnik,flashsparse,dtc}` for real-kernel examples.

---

## 7. External SOTA kernels

Vendored repos are gitignored and fetched on demand:

```bash
bash scripts/fetch_external.sh          # sputnik, FlashSparse, DTC-SpMM into external/
bash scripts/build_flashsparse.sh       # FlashSparse extensions (sm_120)
bash scripts/build_dtc.sh               # DTC-SpMM (applies scripts/patches/dtc_weighted.patch)
```

Notes: they all compile for **sm_120 (Blackwell)**. Sputnik uses a minimal glog shim
(`cpp/third_party/glog_shim`) because its expected glog version is source-incompatible with a
modern one. DTC-SpMM's public wrapper hardcodes unit weights (GNN aggregation); a committed
patch exposes a `run_DTCSpMM_weighted` entry so it computes a true weighted SpMM — its per-
nonzero values are reconstructed into DTC's blocked layout from the preprocess outputs.

> **Lesson (recorded so it doesn't recur):** FlashSparse and DTC-SpMM are *both* general
> weighted SpMM kernels, even though their GNN examples / wrappers default to unit weights.
> Classify a kernel by reading its compute (does it multiply by a value array?), not its API
> surface.
