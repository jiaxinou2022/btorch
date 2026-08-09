# Persistent SNN Pipeline plan 4 results

## Test setup

- GPU: NVIDIA GeForce RTX 5090
- Dataset: FlyBrain, batch size 1, 128 timesteps
- Fixed pipeline controls: one dedicated warp, one helper warp, ticket chunk 1,
  static waves 0
- CUDA Event component values: medians of 30 samples

## Implementation

The pipeline now supports a two-ended task queue:

- HIGH rows reserve forward-growing 1024-edge fragments and run on a full warp.
- LOW rows reserve one neuron descriptor from the back of the same allocation.
- A LOW claim packs `32 / subwarp_size` consecutive descriptors into one warp.
- Each warp obtains a unique future claim with `atomicAdd`, then waits for the
  claimed descriptors' epoch/ready publication. This avoids both dropped work
  and the head-CAS contention found in the first prototype.
- HIGH and LOW claims alternate according to a compile-time scheduling ratio.
- UPDATE helpers use the same consumer protocol after publishing completion.

Threshold 0 retains the pre-plan-4 pipeline consumer and is the no-binning
baseline.

## Threshold scan

Role 7:1, LOW subwarp 8, and a 1:1 HIGH/LOW schedule were fixed:

| Threshold | Core kernel (ms) | Result vs baseline |
| --------: | ---------------: | -----------------: |
| 0 | 12.596 | baseline |
| 128 | 9.847 | -21.8% |
| 256 | 9.401 | -25.4% |
| 512 | 9.346 / 9.352 | -25.8% |

Threshold 512 won and was used for the remaining scans.

## LOW subwarp scan

| Lanes per LOW neuron | LOW tasks per warp | Core kernel (ms) |
| -------------------: | ------------------: | ---------------: |
| 4 | 8 | 9.363 |
| 8 | 4 | 9.307 |
| 16 | 2 | 10.220 |

Eight lanes remained optimal.

## HIGH/LOW scheduling scan

| HIGH : LOW | Core kernel (ms) |
| ----------: | ---------------: |
| 1:1 | 9.306 |
| 2:1 | 9.224 / 9.225 |
| 4:1 | 9.256 |

The selected schedule is two HIGH fragments per LOW group.

## Role ratio rescan

| UPDATE : dedicated PROP blocks | Core kernel (ms) |
| -----------------------------: | ---------------: |
| 7:1 | 9.223 |
| 3:1 | 9.257 |
| 2:1 | 9.214 |
| 1:1 | 9.240 |

A same-GPU repeat measured 7:1 at 9.232 ms and 2:1 at 9.231 ms. The difference
is not meaningful, so the existing 7:1 default is retained.

## Final timeline

With threshold 512, LOW subwarp 8, HIGH/LOW 2:1, and role 7:1:

| Metric | Median per timestep |
| :----- | ------------------: |
| UPDATE | 17.408 us |
| First publish delay | 2.560 us |
| Consumer startup | 2.304 us |
| Overlap window | 12.544 us |
| Propagation tail | 44.544 us |
| Pipeline total | 62.208 us |

The instrumented plan-3 role-scan total was about 94.2 us, so the binned
pipeline reduces this timeline by about 34%. The production core kernel moves
from 12.596 ms to about 9.225 ms, a 26.8% reduction, and is below the earlier
11.8 ms naive reference.

## Correctness and queue validation

The final default configuration produced:

- zero spike mismatches against the PyTorch reference;
- 148,397 HIGH tasks published and processed;
- 4,656,608 LOW tasks published and processed;
- 518,565,492 classified propagation edges, matching active synapses;
- 1,164,201 LOW claims, or 3.9998 tasks per claim;
- 95 partial final LOW groups across 128 timesteps;
- zero queue overflow.

Final defaults are role 7:1, threshold 512, LOW subwarp 8, HIGH/LOW 2:1,
dedicated/helper warps 1/1, ticket chunk 1, and static waves 0. Threshold 0
remains available as the no-binning fallback.
