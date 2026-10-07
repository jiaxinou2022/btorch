# Figure E Slurm run record

All GPU stages run on the RTX 5090 server through Slurm. Paths below are on
the server unless stated otherwise.

## Completed stages

- Download array: job `36679` (`COMPLETED`).
- CSR preparation array: job `36687`; tasks 1--4 completed. Task 0 exposed
  the multi-matrix Orkut archive and was rerun after selecting the exact
  primary Matrix Market member.
- Orkut CSR preparation retry: job `36692` (`COMPLETED`).
- CSR/GPU-memory/SpMV validation array: job `36693` (`COMPLETED`, all five
  datasets).
- Initial benchmark smoke: job `36694` (`FAILED`). Large-tensor workload
  hashing used float-spaced indices, whose loss of integer precision could
  generate an out-of-bounds CUDA index. The replacement uses integer-only
  sampling and has a regression test at 316,548,962 elements.

## Active pipeline

- Strict five-dataset smoke array: job `36709`, array `0-4%1`.
- Four-dataset firing-rate sweep: job `36710`, array `0-119%4`, with Slurm
  dependency `afterok:36709`.

The smoke task validates Hollywood-2009, vas_stokes_4M, Orkut, uk-2002, and
Queen_4147 at `T=8`. Each task fails unless both requested providers produce
a correctness status beginning with `passed`. The dependent sweep therefore
cannot start after a silent provider error.

At submission time on 2026-10-07, all four GPUs were allocated to other
users' jobs. Job `36709` was pending for priority; no GPU command was run
outside Slurm.

## Server outputs

- Prepared CSR: `/data/zhanghan/suitesparse/processed`
- Validation JSON and smoke CSV: `/data/zhanghan/suitesparse/results`
- Activity parts: `/data/zhanghan/suitesparse/results/activity_parts`
- Slurm logs: `/data/zhanghan/suitesparse/slurm_logs`
