# Figure E: activity scaling on large sparse matrices

This figure extends Figure D without changing its scientific question or
timing contract. It measures end-to-end batch-one FP32 RSNN speedup over
cuSPARSE with CUDA Graph capture across average neuronal firing rates from
0 to 50 Hz, with dense sampling at 0.1--10 Hz.

## Datasets

| Figure dataset | SuiteSparse source | Neurons | Stored nonzeros | Role |
|---|---|---:|---:|---|
| Hollywood-2009 | LAW/hollywood-2009 | 1,139,905 | 113,891,327 | initial large-graph validation |
| vas_stokes_4M | VLSI/vas_stokes_4M | 4,382,246 | 131,577,616 | high-neuron-count matrix |
| Orkut | SNAP/com-Orkut | 3,072,441 | 234,370,166 | large social graph |
| Queen_4147 | Janna/Queen_4147 | 4,147,110 | 316,548,962 | required >300M scale |

`uk-2002` (18,520,486 neurons; 298,113,762 nonzeros) is downloaded and
included in CSR, GPU-memory, and short-window smoke validation. It is not a
planned Figure E panel because a 256-step FP32 input trace plus reference
spike trace alone requires about 37.9 GB, already exceeding a 32 GB RTX 5090
before graph storage and provider workspaces.

## Execution stages

1. Download official Matrix Market archives with a CPU Slurm job.
2. Canonicalize each matrix to source-oriented CSR and precompute its
   transposed CSR, checking the official shape and nonzero count.
3. Construct both layouts on one RTX 5090, record allocated/reserved peak GPU
   memory, and execute one sparse matrix-vector multiply.
4. Run an eight-step two-provider benchmark smoke test.
5. Only after all earlier stages pass, submit the full activity sweep for the
   four figure datasets. Each provider runs in a fresh process to bound peak
   memory while reusing the baseline provider's calibrated input amplitude.

All GPU commands are submitted through Slurm. The full sweep uses `B=1`,
`T=256`, `dt=1 ms`, three deterministic workload seeds, ten firing-rate
levels (0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, and 50 Hz), ten warmups, and 20
timed repetitions per seed/provider/rate point.
