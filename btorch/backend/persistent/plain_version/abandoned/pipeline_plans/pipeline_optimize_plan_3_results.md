# Persistent SNN Pipeline plan 3 results

## Test setup

- GPU: NVIDIA GeForce RTX 5090
- Dataset: FlyBrain, batch size 1, 128 timesteps
- Environment: `ml-py312`
- Fixed consumer settings: one dedicated warp, one helper warp, ticket chunk 1
- Serial references: UPDATE 20.992 us, propagation 70.400 us
- CUDA Event component values are medians of 30 samples

## Static startup scan

The timestamp build produced the following per-timestep medians:

| Static waves | UPDATE (us) | Startup (us) | Overlap (us) | Tail (us) | Total (us) |
| -----------: | ----------: | -----------: | -----------: | --------: | ---------: |
| 0 | 29.952 | 7.168 | 18.944 | 65.280 | 95.232 |
| 1 | 30.208 | 7.168 | 19.200 | 65.024 | 95.488 |
| 2 | 30.464 | 6.912 | 19.712 | 65.280 | 95.488 |
| 4 | 30.464 | 6.912 | 19.456 | 65.280 | 95.488 |

The corresponding core-kernel samples were 12.623, 12.626, 12.672, and
12.607 ms. Repeated wave 0 measurements ranged up to 12.669 ms, so their
sub-0.5% ordering was not reproducible. Static assignment slightly reduced
startup at wave 2/4, but did not reduce pipeline total. The selected setting
is therefore `static_waves=0`, which preserves the dynamic baseline.

## Publication curve

At the selected wave 0 setting, median queue publication ratios were:

| UPDATE blocks complete | Published tasks / final tasks |
| ---------------------: | ----------------------------: |
| 25% | 0.735 |
| 50% | 0.901 |
| 75% | 0.978 |
| 100% | 1.000 |

The median final queue contained 37,725 tasks per timestep. Publication is
front-loaded rather than back-loaded, so the conditional fanout-based UPDATE
ordering experiment was not implemented. It would trade contiguous neuron
state access for task supply that is already available early.

## Role ratio rescan

The best static setting was used for the role scan:

| Role | UPDATE (us) | Startup (us) | Overlap (us) | Tail (us) | Slowdown (us) | Hidden (us) | Exchange | Net (us) | Total (us) |
| :--- | ----------: | -----------: | -----------: | --------: | ------------: | ----------: | -------: | -------: | ---------: |
| 7:1 | 25.600 | 9.984 | 10.752 | 68.608 | 4.608 | 1.792 | 0.389 | -2.816 | 94.208 |
| 3:1 | 25.600 | 9.984 | 10.496 | 68.864 | 4.608 | 1.536 | 0.333 | -3.072 | 94.208 |
| 2:1 | 25.344 | 9.728 | 10.240 | 68.608 | 4.352 | 1.792 | 0.412 | -2.560 | 94.208 |
| 1:1 | 30.208 | 7.168 | 19.200 | 65.280 | 9.216 | 5.120 | 0.556 | -4.096 | 95.488 |

All exchange ratios remain below one. The 7:1, 3:1, and 2:1 timestamp totals
tied, while production core-kernel medians were 12.649, 12.699, and 12.659 ms;
1:1 measured 12.671 ms. Two additional same-GPU comparisons measured 7:1 at
12.646/12.646 ms and 2:1 at 12.653/12.657 ms. The selected default is 7:1.

## Implementation and validation

- Dedicated propagation warps support deterministic startup assignments for
  0, 1, 2, or 4 waves, followed by the unchanged dynamic ticket queue.
- Helpers remain dynamic-only and join after their UPDATE block completes.
- Debug timing records queue tails at 25%, 50%, 75%, and 100% UPDATE-block
  completion, and the benchmark reports publication and exchange metrics.
- FlyBrain correctness passed with zero spike mismatches.
- Debug validation processed 4,805,005 tasks, exactly matching the expected
  4,805,005 published tasks.
- Local targeted tests pass, including accepted and rejected static-wave
  configuration points.

Final defaults are role ratio 7:1, one dedicated warp, one helper warp, ticket
chunk 1, and static waves 0.
