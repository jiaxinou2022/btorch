# Fanout preprocessing experiment (v3)

Numerical results only; no visualization files were generated.

## Configuration

```json
{
  "datasets": [
    "uniform",
    "community",
    "spatial",
    "flybrain"
  ],
  "target_rates_hz": [
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0
  ],
  "calibration_input_rates": [
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05
  ],
  "calibration_steps": 512,
  "phase1_steps": 1024,
  "kernel_steps": 512,
  "synthetic_neurons": 8192,
  "synthetic_degree": 64,
  "recurrent_strength": 0.1,
  "greedy_window": 128,
  "sketch_window": 256,
  "task_sample": 100000,
  "kernel_warmup": 10,
  "kernel_repeats": 30,
  "phase1_atomic_threshold": 0.01,
  "load_cv_tolerance": 0.15,
  "seed": 20260901
}
```

## Phase 1 FlyWire screening

| method                     | family             |   median_atomic_reduction |   median_ci95_low |   max_load_cv_ratio |   incremental_preprocess_ms |   total_preprocess_ms |   temporary_memory_mb | phase1_pass   | selected_for_phase2   | selection_reason   |
|:---------------------------|:-------------------|--------------------------:|------------------:|--------------------:|----------------------------:|----------------------:|----------------------:|:--------------|:----------------------|:-------------------|
| sketch_local_union_w256    | sketch_local_union |                  0.03013  |          0.003477 |              1.271  |                3549         |          3549         |              116.7    | False         | False                 | not_selected       |
| union_w128                 | union              |                  0.01706  |         -0.006737 |              1.041  |                1959         |             1.529e+04 |              116.2    | True          | True                  | threshold_pass     |
| normalized_union_a0.5_w128 | normalized_union   |                  0.01653  |         -0.009328 |              1.02   |                2127         |             1.546e+04 |              116.2    | True          | True                  | threshold_pass     |
| union_w128_contention_seed | union              |                  0.01642  |         -0.009701 |              1.04   |                2081         |             1.542e+04 |              116.2    | True          | False                 | not_selected       |
| weighted_union_log_w128    | weighted_union     |                  0.0137   |         -0.009417 |              1.042  |                2001         |             1.534e+04 |              116.2    | True          | True                  | threshold_pass     |
| weighted_union_sqrt_w128   | weighted_union     |                  0.01344  |         -0.00692  |              1.041  |                1992         |             1.533e+04 |              116.2    | True          | False                 | not_selected       |
| union_w128_first           | union              |                  0.01179  |         -0.008751 |              1.036  |                2009         |             1.534e+04 |              116.2    | True          | False                 | not_selected       |
| dominant                   | dominant           |                  0        |          0        |              1      |                   1.334e+04 |             1.334e+04 |                4.372  | False         | False                 | not_selected       |
| sketch128                  | sketch             |                 -0.002239 |         -0.02736  |              1.133  |                  35.95      |            35.95      |                6.368  | False         | False                 | not_selected       |
| top4                       | topk               |                 -0.003363 |         -0.02402  |              0.5381 |                4359         |          4359         |               14.32   | False         | False                 | not_selected       |
| identity                   | identity           |                 -0.00392  |         -0.02822  |              0.26   |                   0.08348   |             0.08348   |                0.5298 | False         | False                 | not_selected       |

The exact-global greedy variant is intentionally excluded: its quadratic candidate scan is not an engineering candidate for 138k FlyWire neurons. All reported greedy methods use explicit bounded windows.

## Phase 2 real persistent shared-hash kernel

|   posterior_population_hz | method                     |   time_per_step_us |   median_speedup |   speedup_p25 |   speedup_p75 |   active_block_tasks |   active_block_task_reduction |   edges_processed |   spike_mismatch_rate_vs_dominant |
|--------------------------:|:---------------------------|-------------------:|-----------------:|--------------:|--------------:|---------------------:|------------------------------:|------------------:|----------------------------------:|
|                    0.4969 | dominant                   |              16.71 |        0         |     0         |     0         |                35012 |                     0         |           3950595 |                         0         |
|                    0.4969 | normalized_union_a0.5_w128 |              16.75 |       -0.002581  |    -0.01113   |     0.00107   |                34977 |                     0.0009997 |           3950595 |                         0         |
|                    0.4969 | union_w128                 |              16.71 |       -0.0001103 |    -0.008768  |     0.006636  |                34996 |                     0.000457  |           3950595 |                         0         |
|                    0.4969 | weighted_union_log_w128    |              16.66 |        0.002986  |    -0.007549  |     0.003672  |                35003 |                     0.0002571 |           3950595 |                         0         |
|                    1.003  | dominant                   |              18.01 |        0         |     0         |     0         |                70124 |                     0         |           8522385 |                         0         |
|                    1.003  | normalized_union_a0.5_w128 |              18.14 |       -0.007509  |    -0.01114   |    -0.002867  |                70113 |                     0.0001569 |           8522385 |                         0         |
|                    1.003  | union_w128                 |              18.04 |       -0.001663  |    -0.006222  |     0.004368  |                70109 |                     0.0002139 |           8522385 |                         0         |
|                    1.003  | weighted_union_log_w128    |              18.11 |       -0.005502  |    -0.008927  |     0.002845  |                70121 |                     4.278e-05 |           8522385 |                         0         |
|                    1.987  | dominant                   |              19.97 |        0         |     0         |     0         |               136828 |                     0         |          17509478 |                         0         |
|                    1.987  | normalized_union_a0.5_w128 |              19.93 |        0.002004  |    -0.005218  |     0.008337  |               136858 |                    -0.0002193 |          17509478 |                         0         |
|                    1.987  | union_w128                 |              20.02 |       -0.002103  |    -0.009397  |     0.002356  |               136873 |                    -0.0003289 |          17509478 |                         0         |
|                    1.987  | weighted_union_log_w128    |              19.87 |        0.005298  |    -0.001831  |     0.007672  |               136876 |                    -0.0003508 |          17509478 |                         0         |
|                    4.751  | dominant                   |              23.44 |        0         |     0         |     0         |               313426 |                     0         |          42646655 |                         0         |
|                    4.751  | normalized_union_a0.5_w128 |              23.42 |        0.0008846 |    -0.001289  |     0.005465  |               313160 |                     0.0008487 |          42646655 |                         0         |
|                    4.751  | union_w128                 |              23.38 |        0.002726  |     0.0002796 |     0.006915  |               313247 |                     0.0005711 |          42646655 |                         0         |
|                    4.751  | weighted_union_log_w128    |              23.41 |        0.001363  |    -0.002096  |     0.00437   |               313213 |                     0.0006796 |          42646655 |                         0         |
|                   12.63   | dominant                   |              37.38 |        0         |     0         |     0         |               670969 |                     0         |         187625702 |                         0         |
|                   12.63   | normalized_union_a0.5_w128 |              37.83 |       -0.01192   |    -0.05042   |     0.0006175 |               654613 |                     0.02438   |         187626204 |                         1.558e-05 |
|                   12.63   | union_w128                 |              37.59 |       -0.005688  |    -0.03143   |     0.003967  |               655100 |                     0.02365   |         187625702 |                         0         |
|                   12.63   | weighted_union_log_w128    |              38.45 |       -0.02793   |    -0.04395   |    -0.01333   |               654989 |                     0.02382   |         187625702 |                         0         |
|                   21.8    | dominant                   |              46.23 |        0         |     0         |     0         |              1064348 |                     0         |         277501682 |                         0         |
|                   21.8    | normalized_union_a0.5_w128 |              46.85 |       -0.01312   |    -0.02893   |     0.000411  |              1049386 |                     0.01406   |         277501682 |                         0         |
|                   21.8    | union_w128                 |              46.79 |       -0.01189   |    -0.02014   |     0.001609  |              1049370 |                     0.01407   |         277501682 |                         0         |
|                   21.8    | weighted_union_log_w128    |              46.36 |       -0.002782  |    -0.01529   |     0.005582  |              1049079 |                     0.01435   |         277501682 |                         0         |
|                   50.1    | dominant                   |              74.34 |        0         |     0         |     0         |              1759378 |                     0         |         533322292 |                         0         |
|                   50.1    | normalized_union_a0.5_w128 |              74.72 |       -0.005031  |    -0.02806   |    -0.001702  |              1751432 |                     0.004516  |         533322292 |                         2.818e-08 |
|                   50.1    | union_w128                 |              74.74 |       -0.005392  |    -0.02008   |     0.0003156 |              1751784 |                     0.004316  |         533322292 |                         2.818e-08 |
|                   50.1    | weighted_union_log_w128    |              75.36 |       -0.01358   |    -0.03034   |    -0.003376  |              1751539 |                     0.004456  |         533322292 |                         2.818e-08 |

## Cross-rate kernel decision

| method                     |   median_speedup |   minimum_speedup |   positive_conditions |   conditions |
|:---------------------------|-----------------:|------------------:|----------------------:|-------------:|
| normalized_union_a0.5_w128 |        -0.005031 |          -0.01312 |                     2 |            7 |
| union_w128                 |        -0.002103 |          -0.01189 |                     1 |            7 |
| weighted_union_log_w128    |        -0.002782 |          -0.02793 |                     3 |            7 |

Adoption requires a stable real-kernel improvement, acceptable preprocessing cost, and no material load-balance regression. Nsight counters are not collected automatically because profiling changes the execution mode; the timed rows use the uninstrumented production hash specialization.

## Decision

The strongest structural candidate was `sketch_local_union_w256` at 3.01% median estimated atomic reduction, but its worst Load-CV ratio was 1.271, so it failed the 15% load-balance gate.

The best isolated real-kernel result was `weighted_union_log_w128` at 1.99 Hz with 0.53% speedup. The best cross-rate median was `union_w128` at -0.21%. Neither reaches the 1% adoption threshold or remains positive across rates.

Decision: reject every tested v3 fanout candidate for the current persistent shared-hash kernel. Structural atomic headroom did not translate into runtime speedup. Retain the dominant baseline.

Across the selected methods and seven conditions, the descriptive Pearson association between Phase-1 atomic reduction and kernel speedup was -0.534; the association between active-BlockTask reduction and speedup was -0.705. These pooled correlations are diagnostic rather than causal, but they confirm that the structural proxies do not predict runtime benefit here.

The maximum reordered-vs-baseline spike mismatch rate was 1.56e-05; this is reported because floating-point atomic accumulation order can perturb threshold crossings.

Phase-1 atomic counts are ratio-of-means estimates from uniformly sampled active BlockTasks. Phase-2 speedups use randomized interleaved CUDA Event samples against the dominant baseline.
