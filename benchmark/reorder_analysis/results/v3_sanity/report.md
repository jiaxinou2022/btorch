# Fanout preprocessing experiment (v3)

Numerical results only; no visualization files were generated.

## Configuration

```json
{
  "datasets": [
    "uniform",
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
  "calibration_steps": 128,
  "phase1_steps": 256,
  "kernel_steps": 128,
  "synthetic_neurons": 2048,
  "synthetic_degree": 32,
  "recurrent_strength": 0.1,
  "greedy_window": 64,
  "sketch_window": 128,
  "task_sample": 10000,
  "kernel_warmup": 5,
  "kernel_repeats": 10,
  "phase1_atomic_threshold": 0.01,
  "load_cv_tolerance": 0.15,
  "seed": 20260901
}
```

## Phase 1 FlyWire screening

| method                    | family             |   median_atomic_reduction |   median_ci95_low |   max_load_cv_ratio |   preprocess_ms |   temporary_memory_mb | phase1_pass   | selected_for_phase2   |
|:--------------------------|:-------------------|--------------------------:|------------------:|--------------------:|----------------:|----------------------:|:--------------|:----------------------|
| dominant                  | dominant           |                  0        |           0       |              1      |       1.489e+04 |                4.373  | False         | False                 |
| union_w64_contention_seed | union              |                 -0.001575 |          -0.06931 |              1.015  |    1251         |              116.2    | False         | True                  |
| sketch_local_union_w128   | sketch_local_union |                 -0.006594 |          -0.07683 |              1.213  |    2065         |              116.7    | False         | True                  |
| normalized_union_a0.5_w64 | normalized_union   |                 -0.01498  |          -0.06303 |              1.006  |    1297         |              116.2    | False         | False                 |
| weighted_union_log_w64    | weighted_union     |                 -0.01736  |          -0.08036 |              1.016  |    1134         |              116.2    | False         | False                 |
| union_w64                 | union              |                 -0.02124  |          -0.101   |              1.017  |    1141         |              116.2    | False         | False                 |
| top4                      | topk               |                 -0.02258  |          -0.1012  |              0.5381 |    4441         |               14.32   | False         | False                 |
| weighted_union_sqrt_w64   | weighted_union     |                 -0.0266   |          -0.08623 |              1.016  |    1184         |              116.2    | False         | False                 |
| union_w64_first           | union              |                 -0.02823  |          -0.07309 |              1.012  |    1281         |              116.2    | False         | False                 |
| sketch128                 | sketch             |                 -0.05544  |          -0.1294  |              1.133  |      46.33      |                6.368  | False         | False                 |
| identity                  | identity           |                 -0.0775   |          -0.1693  |              0.26   |       0.06737   |                0.5298 | False         | False                 |

The exact-global greedy variant is intentionally excluded: its quadratic candidate scan is not an engineering candidate for 138k FlyWire neurons. All reported greedy methods use explicit bounded windows.

## Phase 2 real persistent shared-hash kernel

|   posterior_population_hz | method                    |   time_per_step_us |   median_speedup |   speedup_p25 |   speedup_p75 |   active_block_tasks |   edges_processed |   spike_mismatch_rate_vs_dominant |
|--------------------------:|:--------------------------|-------------------:|-----------------:|--------------:|--------------:|---------------------:|------------------:|----------------------------------:|
|                    0.4988 | dominant                  |              22.16 |         0        |      0        |      0        |                 8783 |            955665 |                                 0 |
|                    0.4988 | sketch_local_union_w128   |              23    |        -0.03329  |     -0.06716  |      0.03508  |                 8776 |            955665 |                                 0 |
|                    0.4988 | union_w64_contention_seed |              22.17 |        -0.003868 |     -0.04651  |      0.04802  |                 8796 |            955665 |                                 0 |
|                    1.007  | dominant                  |              20.63 |         0        |      0        |      0        |                17591 |           2137131 |                                 0 |
|                    1.007  | sketch_local_union_w128   |              20.21 |         0.01625  |      0.001593 |      0.02373  |                17582 |           2137131 |                                 0 |
|                    1.007  | union_w64_contention_seed |              20.06 |         0.01296  |     -0.007767 |      0.03246  |                17609 |           2137131 |                                 0 |
|                    1.977  | dominant                  |              23.04 |         0        |      0        |      0        |                34039 |           4347282 |                                 0 |
|                    1.977  | sketch_local_union_w128   |              22.34 |         0.00359  |     -0.01381  |      0.03887  |                34005 |           4347282 |                                 0 |
|                    1.977  | union_w64_contention_seed |              23.12 |        -0.01476  |     -0.02061  |      0.01217  |                34062 |           4347282 |                                 0 |
|                    4.781  | dominant                  |              24.67 |         0        |      0        |      0        |                78802 |          10660996 |                                 0 |
|                    4.781  | sketch_local_union_w128   |              24.66 |         0.01843  |      0.005599 |      0.02368  |                78774 |          10660996 |                                 0 |
|                    4.781  | union_w64_contention_seed |              25.43 |        -0.01075  |     -0.05413  |      0.01464  |                78770 |          10660996 |                                 0 |
|                   11.43   | dominant                  |              35.42 |         0        |      0        |      0        |               158912 |          37853169 |                                 0 |
|                   11.43   | sketch_local_union_w128   |              34.8  |         0.01074  |     -0.01615  |      0.02981  |               158181 |          37853169 |                                 0 |
|                   11.43   | union_w64_contention_seed |              36.26 |        -0.009632 |     -0.03997  |      0.009615 |               158111 |          37853169 |                                 0 |
|                   21.23   | dominant                  |              45.47 |         0        |      0        |      0        |               263224 |          64910307 |                                 0 |
|                   21.23   | sketch_local_union_w128   |              45.55 |         0.003422 |     -0.01632  |      0.007854 |               261493 |          64910307 |                                 0 |
|                   21.23   | union_w64_contention_seed |              46.14 |        -0.005539 |     -0.02839  |      0.003108 |               262070 |          64910307 |                                 0 |
|                   49.87   | dominant                  |              73.64 |         0        |      0        |      0        |               439625 |         130618695 |                                 0 |
|                   49.87   | sketch_local_union_w128   |              73.6  |         0.00734  |     -0.007582 |      0.01699  |               438954 |         130618695 |                                 0 |
|                   49.87   | union_w64_contention_seed |              74.6  |        -0.00556  |     -0.01395  |      0.003811 |               438930 |         130618695 |                                 0 |

## Cross-rate kernel decision

| method                    |   median_speedup |   minimum_speedup |   positive_conditions |   conditions |
|:--------------------------|-----------------:|------------------:|----------------------:|-------------:|
| sketch_local_union_w128   |          0.00734 |          -0.03329 |                     6 |            7 |
| union_w64_contention_seed |         -0.00556 |          -0.01476 |                     1 |            7 |

Adoption requires a stable real-kernel improvement, acceptable preprocessing cost, and no material load-balance regression. Nsight counters are not collected automatically because profiling changes the execution mode; the timed rows use the uninstrumented production hash specialization.

Phase-1 atomic counts are ratio-of-means estimates from uniformly sampled active BlockTasks. Phase-2 speedups use randomized interleaved CUDA Event samples against the dominant baseline.
