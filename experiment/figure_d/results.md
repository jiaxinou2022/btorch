# Figure D results

The activity sweep was measured on NVIDIA GeForce RTX 5090 GPUs for four
connectivity regimes: FlyBrain (138,639 neurons; 15,091,983 synapses), MICRONS
mm³ (60,048; 6,965,260), a 0.005-scale macaque multi-area model (20,649;
21,960,982), and a uniform fixed-fanout graph (131,072; 16,777,216). All
providers executed the same batch-one, 256-timestep FP32 recurrent LIF and
ExponentialPSC workload. Provider preprocessing and CUDA Graph capture were
excluded from timing.

Graph weights were normalized to a mean absolute weighted recurrent fanout of
approximately 0.15. FlyBrain retains every signed edge weight and uses the
global multiplier `0.15 / 393.0562438964844`; the other graph loaders use a
weight scale of 0.15. The macaque model uses `n_scaling=0.005` and
`k_scaling=1.0`.

Each point is the median speedup across three independently shifted external
input traces. Because there are exactly three seeds, the plotted median and
minimum--maximum error bar expose all three seed-level observations. Each
seed-level latency is itself the median of 20 CUDA timing repetitions after 10
warmups. Speedup is matched within dataset, target activity, and input seed,
using cuSPARSE SpMV captured in a CUDA Graph as the 1× baseline.

All 360 provider/workload rows passed the benchmark correctness criterion. The
maximum spike mismatch rate was 0.0536%, below the configured 0.1% tolerance.
At the lowest activity point, PaceRSNN reached median speedups of 9.03× on
FlyBrain, 5.09× on MICRONS, 6.51× on the macaque model, and 8.90× on the
uniform graph. At approximately 20% activity, the corresponding speedups were
2.59×, 2.67×, 2.72×, and 2.36×. The repeated trend supports the intended
conclusion that PaceRSNN's advantage is largest in sparse-activity regimes and
narrows as active work increases.

The x-axis always uses measured activity. Of 72 calibrated workloads, 61 met
the relative tolerance and 11 were retained as stable `best_effort` points.
The largest discontinuities were MICRONS seed 0 at the requested 20% point
(11.21% measured) and the uniform graph at the requested 10% point
(5.94%--12.65% measured across seeds). Expanding the MICRONS calibration bound
from 100 to 500 did not change that stable solution. These observations are
not relabeled as their requested rates; both requested and measured values and
the calibration status remain in the raw and summary tables.

## Suggested caption

**Performance across neuronal activity levels.** End-to-end RSNN speedup over
cuSPARSE SpMV with CUDA Graph capture on four connectivity regimes and NVIDIA
GeForce RTX 5090 GPUs. The x-axis reports measured rather than requested firing
activity. Points show medians across three workload seeds and error bars span
the seed-level minimum and maximum; every seed-level latency is the median of
20 timed runs. All methods share the connectivity, input trace, initial state,
and FP32 dynamics (`B=1`, `T=256`); preprocessing and graph capture are
excluded. The horizontal dotted line marks the 1× cuSPARSE baseline.
