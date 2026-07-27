# Warp-specialized descriptor mailbox experiment

> Mode 3 in this report is the retained old B3 implementation. The corrected
> mode 4 experiment is documented in `warp_b3_clean_experiment.md`.

## Scope

This experiment implements the first three gates from `warp_spec_plan.md` in
`persistent_snn_spike_block_warp_spec_kernel.cu`. The original block kernel is
unchanged.

- B0: the original eight-warp block/hash kernel.
- B1: warp 0 is idle during ordinary block tasks; warps 1–7 retain the direct
  load/hash/flush path.
- B2: warp 0 distributes task descriptors to seven fixed consumer mailboxes.
  Consumers retain the direct load/hash/flush path.
- B3: producer lanes 0–6 synchronously copy 64-edge physically contiguous
  chunks into fixed mailboxes. Consumers locate the owner neuron from the
  edge's relative position and retain hash state across chunks.

Set `BTORCH_WARP_SPEC_MODE` to `0`, `1`, `2`, or `3` before loading the
extension. Mode 0 remains the default and selects the original source file.

## Implementation

B2 gives every consumer a fixed task descriptor and two monotonic epochs.
Producer lanes 0–6 each serve one consumer, obtain unique tasks through the
existing global work counter, and publish descriptors with block-scoped
fences. Consumers never compete for a mailbox and no shared atomic CAS polling
is used.

The producer has no hash table. All experimental modes therefore use seven
512-entry hash tables rather than eight.

B3 only stages tasks whose active owner mask is one contiguous run and whose
descriptor uses the explicit block-edge-budget format. Other tasks use the B2
direct path. This keeps staged input physically contiguous and gives it a 100%
active-edge ratio without making the producer perform sparse gather mapping.

## Correctness

Hardware: NVIDIA GeForce RTX 5090 (SM 12.0).

Dataset and configuration:

```text
flybrain, batch=1, timesteps=8
block_edge_budget=256, long_segment_size=512
hash=(aggregation=512, capacity=512, probes=4, min_edges=256)
reorder=global_similarity
```

All three modes passed the PyTorch comparison with zero spike mismatches. B2
and B3 completed a forced 340-block cooperative launch, confirming two
resident blocks per each of the GPU's 170 SMs.

Compiled kernel resources:

| Version | Registers/thread | Static shared/block | Local stack/thread |
| --- | ---: | ---: | ---: |
| B1 | 48 | 38,912 B | 0 B |
| B2 | 46 | 39,000 B | 0 B |
| B3 | 56 | 42,700 B | 0 B |

## Performance

The server was shared with other GPU processes, so absolute times are
contended. B1/B2 were run in an interleaved order to make their relative
comparison useful. Each run used 128 timesteps, 10 warmups, and 50 repeats.

| Version | Runs (us/timestep) | Mean (us/timestep) | Relative |
| --- | --- | ---: | ---: |
| B0 | 140.944 | 140.944 | baseline |
| B1 | 144.381, 144.046 | 144.213 | +2.32% vs B0 |
| B2 | 146.284, 146.135 | 146.210 | +1.38% vs B1 |

A later interleaved B2/B3 comparison produced:

| Version | Runs (us/timestep) | Mean (us/timestep) | Relative |
| --- | --- | ---: | ---: |
| B2 | 146.174, 146.252 | 146.213 | baseline |
| B3 | 200.431, 200.039 | 200.235 | +36.95% vs B2 |

An 8-timestep smoke benchmark measured B1 at 69.588 us/timestep and the final
B2 at 72.148 us/timestep (+3.68%).

## Decision

B2 stays within the plan's maximum tolerated 3% regression relative to B1, so
the fixed descriptor ownership gate passes. B3 is 36.95% slower than B2 and
fails the synchronous-mailbox gate.

The first centralized scheduler used a lane-0 scan and a seven-entry local
epoch array. It added 32 B/thread of local stack and made B2 15.71% slower than
B1. Mapping producer lanes 0–6 directly to consumers removed the scan and
local stack, reducing the final protocol overhead to 1.38%.

The unrestricted first B3 prototype loaded the full static 32-neuron block and
measured 1109 us/timestep. Restricting the physical span to the first and last
active owner reduced it to 664 us/timestep, but gaps still caused excessive
loads. The final version stages only contiguous active-owner runs and falls
back for scattered masks.

The final B3 remains slow because one producer lane serially copies each
64-edge mailbox and eligible contiguous-run tasks do not cover enough work to
amortize mailbox synchronization. Four-lane producer subgroups were also
tested, but mixed staged/direct task durations could not safely drain the
subwarp protocol. Per the plan, the experiment stops before async copy or TMA
because B3 itself exceeds the 5% regression limit.
