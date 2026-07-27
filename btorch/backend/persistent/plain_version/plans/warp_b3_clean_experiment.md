# B3-clean continuous-run mailbox experiment

## Implementation

The corrected B3 implementation is compile-time mode 4:

```text
BTORCH_WARP_SPEC_MODE=4
```

It preserves mode 3 as the old B3 comparison. B3-clean changes the pipeline
as follows:

- Lane 0 schedules descriptors and scans all seven consumers.
- The full producer warp cooperatively copies one 128-edge chunk.
- Tasks are staged only for a physically contiguous active-owner run with
  256–512 edges and an explicit block-edge-budget descriptor.
- Staged consumers do not build owner prefixes or perform owner searches.
- Hash state spans all chunks and flushes once after the complete task.
- Task and mailbox epochs use block-scope release/acquire operations.
- Direct tasks retain the B2 consumer path.

Mode 4 additionally exposes coverage, task-size, chunk, active-consumer, and
producer-idle counters in stats builds. Legacy 23-column stats remain accepted
by the benchmark summarizer; current builds return 44 columns.

## Correctness

Hardware: NVIDIA GeForce RTX 5090, compute capability 12.0.

The following checks passed:

- FlyBrain at 1 and 8 timesteps: zero spike mismatches.
- Nine spike-block CUDA tests with hash and block budget explicitly enabled.
- Seven block-stat summary tests.
- Produced chunks equal consumed chunks: 748 in the sampled stats run.
- B2 and B3-clean have identical dense-reference mismatch counts and maxima at
  32 and 128 timesteps. The longer recurrent trajectory amplifies the same
  floating-point atomic-order difference in both modes.

## Resources

| Version | Registers/thread | Shared/block | Local stack | Blocks/SM |
| --- | ---: | ---: | ---: | ---: |
| B2 | 48 | 39,000 B | 0 B | 2 |
| B3-old | 56 | 42,700 B | 0 B | 2 |
| B3-clean | 46 | 46,480 B | 0 B | 2 |

B3-clean completed a forced 340-block cooperative launch on 170 SMs.

## Performance

Configuration:

```text
dataset=flybrain
batch=1
timesteps=128
event_rate=0.02
warmup=10
repeat=50
block_edge_budget=256
long_segment_size=512
hash=(aggregation=512, capacity=512, probes=4, min_edges=256)
reorder=global_similarity
```

The shared server was benchmarked in forward and reverse interleaved order:

| Version | Runs (us/timestep) | Mean | vs B2 |
| --- | --- | ---: | ---: |
| B2 | 168.056, 168.165 | 168.111 | baseline |
| B3-old | 205.117, 205.321 | 205.219 | +22.07% |
| B3-clean | 172.505, 172.333 | 172.419 | +2.56% |

The engineering corrections recover about 16.0% relative to B3-old and bring
the synchronous clean pipeline inside the plan's 5% performance gate.

## Coverage and pipeline statistics

The representative eight-timestep stats run reported:

| Metric | Value |
| --- | ---: |
| Ordinary tasks | 52,272 |
| Ordinary edges | 19,088,423 |
| Eligible/staged tasks | 194 |
| Eligible/staged edges | 92,516 |
| Eligible task coverage | 0.371% |
| Staged edge coverage | 0.485% |
| 256–383 edge tasks | 28 |
| 384–511 edge tasks | 14 |
| 512-edge tasks | 152 |
| Three-chunk tasks | 28 |
| Four-chunk tasks | 166 |
| Maximum active consumers | 7 |
| Average active consumers | 3.71 |
| Producer idle loops | 39,397 |

Coverage sweep:

| Event rate | Reorder | Eligible task | Staged edge | Avg active |
| ---: | --- | ---: | ---: | ---: |
| 0.005 | global similarity | 0.495% | 0.670% | 2.89 |
| 0.020 | global similarity | 0.477% | 0.633% | 2.86 |
| 0.100 | global similarity | 0.354% | 0.444% | 3.07 |
| 0.020 | identity | 0.0065% | 0.0071% | 3.63 |

All cases reached seven simultaneous active consumers, but staged edge
coverage remained below one percent.

## Decision

B3-clean passes correctness, resource, concurrency, and the synchronous
performance gate. It fails the plan's 20% staged-edge coverage gate by a large
margin across every tested FlyBrain activity rate and reorder.

The block pipeline therefore stops before async copy or TMA. Async copy could
only affect less than one percent of ordinary edges in this workload and
cannot justify its complexity. A future pipeline experiment should target
long segments or introduce a layout/preprocessing change that materially
increases physically contiguous eligible coverage.
