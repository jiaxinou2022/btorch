# SpGEMM (MH-SpGEMM) and SpMSpV (VDHA)

Two algorithms added on top of the SpMM pillar, each on its own operation target. Both are
standalone CUDA kernels reached through the out-of-tree entry-point registry, so they land
on the shared leaderboard with no change to the core tree.

Hierarchy (`platform / algorithm / target / variant`):

```
cuda / mh_spgemm / spgemm / csr     C = A·Aᵀ   fp32          (fp16 disabled — see below)
cuda / vdha      / spmspv / csc     y = A·x    fp32 · fp16   (x sparse)
```

Both build **one shared library per value type** (`scripts/build_{mh_spgemm,vdha}.sh` emit
`..._fp32.so` / `..._fp16.so`); the provider loads the `.so` matching the requested
precision. VDHA's fp16 stores matrix values as `__half` (halving value-memory traffic) but
**accumulates in fp32** — clean and correct. MH-SpGEMM's fp16 *builds* (`VALUE_TYPE=__half`),
but the upstream kernel is fp64-centric: its shared-memory hash tables and atomics assume a
wide value type and the `__half` path **faults (aborts) on graphs that hit the
global-memory-pool code path** (e.g. `fly_larva`). So MH-SpGEMM fp16 is **not advertised** —
fp32 is solid; the fp16 lib can still be built (`MH_DTYPES="float half"`) for experimentation.

`platform = cuda` because both are compiled `.cu` kernels invoked via a ctypes C-ABI (the
same slot the SpMM taxonomy fills with `torch` / `jax` / `cupy` / `cpp`; here the honest
runtime is the CUDA toolkit itself). `algorithm = provider`.

---

## Why these two targets

- **SpGEMM `C = A·Aᵀ`** — two-hop / common-neighbour / Markov-expansion structure over a
  connectome. It was already a first-class case type (`SpgemmCase`) with an fp64 reference
  in the C++ harness, but had no accelerated provider path until now.
- **SpMSpV `y = A·x`, x sparse** — the event-driven primitive: `x` = the set of currently
  firing neurons, `A` = the weighted connectivity, `y` = accumulated synaptic input. Only
  firing columns are touched. This is exactly the inner operator of the RSNN pillar viewed
  as a single spike-delivery step, and the operation VDHA is built for. It is a **new
  target** (`SpMSpVCase`, `SPMSPV_DEFAULTS`, `discover_spmspv`, `external/test_spmspv.py`).

---

## MH-SpGEMM (Yang et al.)

Efficient SpGEMM on GPUs via *masking + hashing cooperative optimization*. Its value type is
a build parameter (`VALUE_TYPE`), so we ship fp32 and fp16 libraries. Vendored from
`~/src/MH-SpGEMM` into `external/MH-SpGEMM` (gitignored; `scripts/fetch_external.sh`).

**Integration.** `MH_spgemm(A, B, C)` is a genuine general `C = A·B` — only `B` is tiled and
masked; `A` is read purely as the left operand (verified against the source). So we get
`A·Aᵀ` by passing `B = transpose(A)` with **no macro edits and no `A==B` assumption**. The
C-ABI wrapper (`cpp/baselines/mh_spgemm/mh_spgemm_c.cu`) copies the `MH_spgemm` orchestration
out of the upstream `main.cu` (which also defines `main()` and cannot be linked), uploads
`A` and `B = Aᵀ` once, and times the full pipeline (mem-alloc + mask + symbolic + numeric)
as MH-SpGEMM's own phase-summed runtime — matching how the upstream harness reports it. The
result `C` is copied to host as a CSR only for oracle validation.

**Baseline.** `cupy.cusparse` runs `A @ Aᵀ` via cuSPARSE SpGEMM at fp32 and fp16, sharing a
leaderboard cell with MH-SpGEMM at the same precision (precision is folded into the cell), so
they are ranked head-to-head. Note fp16 SpGEMM accumulates in half, so its long high-degree
row sums can lose accuracy or overflow — the oracle flags any such case `status="incorrect"`.

**Correctness.** Every `C` is compared to `scipy` `A@Aᵀ` in fp64 (`precision.validate_spgemm`)
by the *sparse* difference (tolerant of extra structural zeros), so a fast-but-wrong kernel
is tagged `status="incorrect"` and excluded. `--strict-correctness` promotes that to a test
failure (used in the GPU job's gate).

**Build.** `scripts/build_mh_spgemm.sh` → `cpp/build-mh_spgemm/libconnectome_mh_spgemm_{fp32,fp16}.so`
(sm_120 default; wrapper + `CSR.cu Tool.cu Timing.cpp utils.cpp`, `-fopenmp`, no cuSPARSE; one
lib per `VALUE_TYPE`).

---

## VDHA (Li et al.) — reimplemented

Vector-Driven Hash Aggregation for weighted SpMSpV (fp32 · fp16). There is no upstream source; the
kernel (`cpp/baselines/vdha/vdha.cu`) is reimplemented from the paper. It is the
column-selection ("push") formulation: for each nonzero `(c, s)` of `x`, scale CSC column
`c` of `A` by `s` and scatter-accumulate into `y`.

**What is implemented (the three ablation components):**

1. **Hash aggregation** — each CTA owns a private 2048-entry shared-memory hash table keyed
   on row index (empty = −1), open-addressed with a fixed odd probe stride, `atomicCAS` to
   claim a slot, `atomicAdd` to accumulate; a probe-cap fallback goes straight to a global
   `atomicAdd(&y[row], v)`. The table is flushed to `y` in bucket order.
2. **Long-column split** — a column longer than `SPLIT_SIZE = 256` is cut into ≤256-entry
   segments so no CTA is overloaded by a hub column (connectome/graph degree is heavy-tailed).
3. **Segment reorder** — segments are sorted by the row index of their first nonzero, so
   CTAs touching nearby rows run together and their flushes coalesce.

**Deliberate simplifications** (pure-performance refinements, correctness-neutral): the
block-mapped grouping of many short columns per CTA (we give each short column its own CTA),
and the `cp.async` double-buffered fetch pipeline. Both are noted here so the number is read
as "faithful core VDHA," not "every last optimization."

**Measurement.** The vector-processing pass (classify / split / sort) depends only on `x`,
so it is done once host-side in `prepare()` and reported separately; the self-timed
`compute()` is the device aggregation (memset of `y` + kernel), averaged over internal CUDA
events. This isolates the kernel contribution the same way FlashSparse/DTC self-time.

**Baseline.** `cupy.cusparse` runs a dense-vector `cusparseSpMV` over the densified `x` —
the honest, sparsity-insensitive comparison (it touches all of `A` regardless of how few `x`
entries are set), quantifying exactly the work a vector-driven kernel saves. This is the
paper's own cuSPARSE baseline.

**Correctness.** `y` is compared to `scipy` `A @ x_dense` (`precision.validate_spmspv`).

**Build.** `scripts/build_vdha.sh` → `cpp/build-vdha/libconnectome_vdha_{fp32,fp16}.so`
(single `.cu`, `-DVDHA_VAL` selects the stored value type, sm_120 default).

---

## RSNN spike delivery (btorch)

Both kernels also serve as the recurrent operator of the **btorch connectome RSNN** — the
per-timestep `spikes @ W` that propagates spikes through the connectivity (the beNNch *deliver*
phase). This is the `variant` axis of `torch.btorch.rsnn`, exactly where btorch already records
its native SpMV as `variant="native"` (`torch/btorch/delivery.py`, `test_rsnn_deliver.py`):

- **`native`** — btorch's own `SparseConn` torch sparse SpMV (`spikes @ W`), the baseline.
- **`vdha`** — SpMSpV, one sparse spike vector, `y = Wᵀ·s` (`= spikes @ W`).
- **`mh_spgemm`** — SpGEMM, a batch of sparse spike vectors, `Y = Wᵀ·Sᵀ`, using the *general*
  `A·B` path (`cbn_mh_prepare_ab`).

`spikes @ W == Wᵀ·spikesᵀ`, so the kernels prepare with `Wᵀ`. No `EventFixedProb`: this is the
connectome variant, delivering through the real adjacency (the synthetic coba/cuba random wiring
is a separate, brainstate-only track). Each backend self-times over `timesteps` at the operating
firing rate and is validated against a SciPy `spikes @ W` reference; VDHA (SpMSpV) is compared to
`native` at batch 1, MH-SpGEMM (SpGEMM) at batch 32.

Two hard-won constraints on MH-SpGEMM for delivery: its dense-ish B-tiling **faults on empty B
rows** (→ deliver the event-driven **compressed** form `Wᵀ[:, active]·Sᵀ[active,:]`, only
spiking neurons — identical result, no empty rows) and **on fewer than `BLOCK_SIZE` (32) B
columns** (→ batch ≥ 32). VDHA has neither. Even so, MH-SpGEMM still intermittently hits a
**device-side `illegal memory access`** on dense inputs (e.g. `mice_column`) that C++ `catch`
cannot intercept and can *wedge the GPU for the whole node*. So `native` + `vdha` are the
default RSNN delivery variants; **MH-SpGEMM delivery is opt-in** (`RUN_RSNN_MH=1`), excluded
from the strict gate, and isolated in its own process. VDHA is the robust event-driven kernel
here; MH-SpGEMM is the batched-SpGEMM option where it is stable.

## Running

Both are exercised by one GPU job (partition `debug`, sm_120):

```bash
sbatch scripts/bench_mh_vdha.sbatch
```

It fetches + builds both `.so`, installs the two `examples/connectome_bench_{mh_spgemm,vdha}`
provider packages, runs a **strict correctness gate** on `mice_column_v1`, then sweeps the
connectome corpus:

- SpGEMM: `external/test_spgemm.py` + `cupy/cusparse/test_spgemm.py`, `--precisions fp32,fp16`
- SpMSpV: `external/test_spmspv.py` + `cupy/cusparse/test_spmspv.py`, `--precisions fp32,fp16`

The strict correctness gate runs at fp32 only; fp16 is exercised in the (non-strict) sweeps,
where reduced-precision inaccuracy is recorded rather than aborting the run.

and ingests both into `results/history.jsonl` → `results/leaderboard.md` / `report.html`.

Metrics: SpGEMM is FLOP-rated on intermediate products (`metrics.flops_spgemm`); SpMSpV on
the products actually touched — `Σ` column-nnz over active `x` indices (`metrics.flops_spmspv`)
— so throughput reflects real work, not `2·nnz(A)`.
