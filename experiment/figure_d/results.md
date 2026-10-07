# Figure D results

The corrected sweep reports average neuronal firing rate in hertz. The
simulation timestep is 1 ms, so the conversion from the closed-loop spike
trace is

```text
average firing rate (Hz) = measured spikes/neuron/timestep × 1000 / dt_ms.
```

The sampled range is 0--50 Hz. Seven of the ten nonuniformly spaced levels
(0.1, 0.2, 0.5, 1, 2, 5, and 10 Hz) concentrate measurements in the requested
0.1--10 Hz regime; 0, 20, and 50 Hz provide the silent and higher-rate
references. The figure uses a symmetric-log x-axis with a 0.1 Hz linear
threshold so it can show the true 0 Hz measurement without compressing the
low-rate region.

Measurements cover FlyBrain (138,639 neurons; 15,091,983 synapses), MICRONS
mm³ (60,048; 6,965,260), a 0.005-scale macaque multi-area model (20,649;
21,960,982), and a uniform fixed-fanout graph (131,072; 16,777,216). All
providers execute the same batch-one, 256-timestep FP32 recurrent LIF and
ExponentialPSC workload. Provider preprocessing and CUDA Graph capture are
excluded from timing.

Low firing rates could not be resolved by varying input amplitude while
holding the original 1% external event probability fixed. The corrected
calibration therefore uses an external event probability equal to the target
spikes/neuron/timestep up to a cap of 1%, followed by a 20-iteration binary
search over input amplitude. Every provider within a workload receives the
same input trace, connectivity, initial state, and recurrent spike trace. The
CSV records the external event rate, timestep, requested rate, and measured
rate explicitly.

Each point is the median speedup across three independently shifted external
input traces. Because there are exactly three seeds, the plotted median and
minimum--maximum error bar expose all three observations. Each seed-level
latency is the median of 20 CUDA timing repetitions after 10 warmups. Speedup
is matched within dataset, requested rate, and input seed, using cuSPARSE SpMV
captured in a CUDA Graph as the 1× baseline.

All 600 provider/workload rows passed the correctness criterion. The maximum
spike mismatch rate was 0.0206%, below the configured 0.1% tolerance. Of 120
calibrated workloads, 115 converged within tolerance; the remaining five are
stable FlyBrain best-effort solutions and are plotted at their measured rates.

At 0.1 Hz measured activity, PaceRSNN reached median speedups of 13.54× on
FlyBrain, 8.22× on MICRONS, 12.76× on the macaque model, and 9.53× on the
uniform graph. At approximately 50 Hz, the corresponding values were 5.30×,
4.30×, 5.66×, and 5.04×. The corrected low-rate sweep therefore preserves the
main conclusion: PaceRSNN benefits most strongly from sparse neuronal
activity, while its advantage narrows as firing rate rises.

## Suggested caption

**Performance across average neuronal firing rates.** End-to-end RSNN speedup
over cuSPARSE SpMV with CUDA Graph capture on four connectivity regimes and
NVIDIA GeForce RTX 5090 GPUs. The x-axis reports the measured average firing
rate in Hz (`dt=1 ms`) on a symmetric-log scale; sampling is concentrated from
0.1 to 10 Hz and spans 0 to 50 Hz. Points show medians across three workload
seeds and error bars span the seed-level minimum and maximum; every seed-level
latency is the median of 20 timed runs. All methods share connectivity, input
trace, initial state, and FP32 dynamics (`B=1`, `T=256`); preprocessing and
graph capture are excluded. The horizontal dotted line marks the 1× cuSPARSE
baseline.
