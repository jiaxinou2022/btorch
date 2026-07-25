# Persistent grid sweep analysis

Workload: `mice_column_v1`, 4,166 neurons, 726,404 graph synapses,
128 timesteps, batch size 1, and input event rate 0.01. The GPU has 170 SMs.
Each 256-thread block contributes two active warps per scheduler, so the five
grid sizes correspond to one through five resident blocks per SM.

## Results

| Variant | Blocks/SM | Latency (ms) | Eligible warps/scheduler | Active warps/scheduler | Barrier stall (cycles) | Barrier stall (%) | Tail (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain | 1 | 4.255 | 0.0222 | 2.000 | 42.7 | 46.0 | 0.0069 |
| plain | 2 | 4.233 | 0.0318 | 4.000 | 72.5 | 53.7 | 0.0138 |
| plain | 3 | 4.328 | 0.0414 | 6.000 | 98.7 | 59.2 | 0.0088 |
| plain | 4 | 4.476 | 0.0495 | 8.000 | 125.0 | 62.9 | 0.0209 |
| plain | 5 | 4.725 | 0.0564 | 9.999 | 151.9 | 65.4 | 0.0149 |
| binning | 1 | 4.405 | 0.0206 | 2.000 | 52.6 | 52.4 | 0.0101 |
| binning | 2 | 4.424 | 0.0314 | 4.000 | 84.4 | 60.4 | 0.0094 |
| binning | 3 | 4.598 | 0.0399 | 6.000 | 110.5 | 61.9 | 0.0207 |
| binning | 4 | 4.839 | 0.0464 | 8.000 | 140.8 | 63.2 | 0.0158 |
| binning | 5 | 5.131 | 0.0526 | 10.000 | 171.9 | 64.0 | 0.0224 |
| spike block | 1 | 5.336 | 0.0205 | 2.000 | 71.9 | 71.2 | 0.0169 |
| spike block | 2 | 5.540 | 0.0309 | 4.000 | 103.4 | 74.2 | 0.0223 |
| spike block | 3 | 6.543 | 0.0399 | 6.000 | 124.8 | 73.1 | 0.0246 |
| spike block | 4 | 8.348 | 0.0460 | 8.000 | 152.1 | 72.1 | 0.0577 |
| spike block | 5 | 8.792 | 0.0522 | 10.000 | 178.8 | 71.4 | 0.0281 |

`tail` is the spread between maximum and minimum active SM cycles, converted
to time. NCU duration is retained in the source CSV but is not mixed with the
independent CUDA-event median latency in this table.

## Findings

The current maximum-occupancy launch policy over-resides this workload. From
one to five blocks per SM, active warps increase fivefold while eligible warps
only increase 2.5--2.6 times. Eligible/active warp density therefore falls by
roughly half, and all configurations remain far below one eligible warp per
scheduler.

Grid-wide synchronization scales poorly. From 170 to 850 blocks, barrier stall
cycles rise 3.55x for plain, 3.27x for binning, and 2.49x for spike block.
CUDA-event latency rises by 11.1%, 16.5%, and 64.8%, respectively. Spike block
is already barrier-dominated at one block per SM (71.2%) and gains no benefit
from additional residency.

The measured final SM tail is small: below 0.023 ms except for the 680-block
spike-block point (0.058 ms). The primary issue is repeated barrier waiting
throughout the kernel, not a conventional final partial-wave tail. All tested
grids are exact multiples of the SM count, and cooperative blocks form one
simultaneously resident wave.

For this workload, use one block per SM for binning and spike block. Plain at
one or two blocks per SM is effectively tied (the CUDA-event difference is
0.5%, while NCU duration differs by 0.1%); prefer one block per SM unless a
larger workload demonstrates a reproducible benefit. The production grid
policy should cap residency by useful task volume instead of always selecting
the occupancy maximum. Reducing grid-wide phases and barriers is higher
priority than increasing occupancy.
