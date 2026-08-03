# Long-Segment Warp Specialization Experiment

## Scope

This experiment follows `warp_segment_plan.md` and replaces only the
long-segment phase of the spike-block kernel. The ordinary block/hash path,
neuron update, task construction, recurrent writeback semantics, and timestep
barriers are unchanged.

The implementation is isolated in:

```text
persistent_snn_spike_block_segment_warp_spec_kernel.cu
```

The original `persistent_snn_spike_block_kernel.cu` was not modified.

Implemented compile-time modes:

```text
S0 = eight-warp direct baseline
S1 = seven-warp direct baseline
S2 = one descriptor producer + seven direct consumers
S3 = synchronous shared-memory transport pipeline
```

S4 bulk async and S5 TMA were not implemented because the S2 and S3
continuation gates failed.

## S3 design

The synchronous pipeline uses:

- producer warp 0 and consumer warps 1--7;
- block-scope release/acquire descriptor publication;
- physically contiguous long-segment edge spans;
- configurable 64/128/256-edge chunks;
- configurable two, three, or four shared transport slots;
- one outstanding chunk per consumer;
- full-warp cooperative global-to-shared copies;
- consumer register buffering before early slot release;
- global atomics only after the shared slot is released;
- direct fallback below the pipeline threshold.

Chunk tokens encode the consumer, task epoch, and chunk index. This prevents
an ABA collision when different consumers reuse one slot with the same local
task epoch.

## Instrumentation

The stats build uses 64 columns and records:

- long tasks and edges;
- full and partial segment distributions;
- eligible tasks and edges;
- produced and consumed chunks;
- producer idle/no-slot/no-consumer events;
- consumer task and chunk waits;
- maximum and time-weighted active consumers;
- slot utilization;
- coarse update, ordinary block, and long-path cycles.

Normal builds do not contain these counters.

## Environment

```text
GPU: NVIDIA GeForce RTX 5090
CUDA architecture: sm_120
environment: micromamba ml-py312
dataset: FlyBrain / FlyWire 783
ordinary block edge budget: 256
long segment size: 512
block hash: aggregation 512, capacity 512, max probe 4
reorder: global_similarity
```

Production timings use one otherwise idle RTX 5090, batch 1, 128 timesteps,
event rate 0.020, 10 warmups, and 50 measured repetitions.

## Precondition statistics

The S0 stats build at one timestep reported:

| Metric | Result |
| --- | ---: |
| Long tasks | 120 |
| Long edges | 45,372 |
| Long recurrent-edge coverage | 29.99% |
| Full 512 segment ratio | 33.33% |
| Average edges per segment | 378.1 |
| Long-path coarse cycle share | 11.80% |

Both plan continuation conditions were met:

```text
long edge coverage >= 15%
long path cycle share >= 10%
```

At eight recurrent timesteps, long-edge coverage increased to 41.79%.

## Correctness

The S3 pipeline passed:

- FlyBrain one and eight timestep PyTorch comparisons;
- zero spike mismatch at eight timesteps;
- 45,356 produced and 45,356 consumed chunks for the selected S3
  configuration;
- all segment-tail boundaries 1, 127, 128, 129, 255, 256, 383, 384, 511,
  and 512;
- task-count drain cases with 1, 2, 7, and 8 full segments;
- the complete CUDA persistent SNN test file: 27 passed;
- block-stat summary tests: 8 passed.

The eight-timestep PSC maximum absolute difference was 0.20996, consistent
with the existing recurrent atomic-order tolerance.

## Performance

### S0--S3 gates

| Mode | Time (us/timestep) | Relative result |
| --- | ---: | ---: |
| S0, 8-warp direct | 68.319 | baseline |
| S1, 7-warp direct | 67.491 | -1.21% vs S0 |
| S2, descriptor + direct | 70.246 | +4.08% vs S1 |
| S3 best sync pipeline | 74.212 | +5.65% vs S2 |

S1 passed its 5% gate. S2 missed its maximum 3% additional-regression gate.
The best S3 result also missed its maximum 5% regression gate.

### S3 parameter sweep

Representative same-GPU results:

| Chunk | Stages | Threshold | Time (us/timestep) |
| ---: | ---: | ---: | ---: |
| 64 | 3 | 256 | 132.859 |
| 128 | 3 | 128 | 126.924 |
| 128 | 3 | 256 | 107.670 |
| 128 | 2 | 256 | 98.429 |
| 128 | 4 | 256 | 97.722 |
| 128 | 3 | 384 | 82.485 |
| 128 | 3 | 512 | 74.290 |
| 128 | 2 | 512 | **74.212** |
| 128 | 4 | 512 | 74.341 |

A 256-edge chunk with three stages does not compile in the selected block
hash configuration: it requires 49,440 bytes of shared memory, above the
48 KiB per-block limit.

Raising the threshold to 512 is the dominant improvement. It pipelines only
full segments and avoids protocol overhead on partial segments, but it still
does not pass the S3 performance gate.

## Pipeline and resource results

For the best S3 configuration (chunk 128, stages 2, threshold 512):

| Metric | Result |
| --- | ---: |
| Pipeline coverage of long edges | 42.37% |
| Maximum active consumers | 7 |
| Time-weighted average active consumers | 4.70 |
| Registers, non-dense kernel | 47 |
| Registers, dense-return kernel | 64 |
| Local stack / spills | 0 / 0 |
| Shared memory per block | 45,324 bytes |
| Residency check | 340 blocks on 170 SMs |

The coverage and maximum-active-consumer gates pass, and two blocks per SM
remain possible. Performance is the failing condition.

## Decision

Stop after S3.

The measured behavior indicates that global atomic scatter does not hide
enough synchronous producer/consumer protocol cost. The best configuration
restricts transport to full 512-edge segments, but remains 5.65% slower than
S2 and 8.63% slower than S0. S2 itself is 4.08% slower than S1.

According to the plan, bulk async and TMA must not be added on top of this
failed synchronous baseline. The independent S0--S3 implementation and its
tests are retained for reproducibility, while the production default remains
the existing block kernel.
