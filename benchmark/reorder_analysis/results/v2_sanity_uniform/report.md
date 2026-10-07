# Fanout-preserving refinement experiment (v2)

Numerical results only; no images or visualization files were generated.
Profile and evaluation traces use different seeds. Oracle orders are built exclusively from the profile trace.

## Configuration

```json
{
  "datasets": [
    "uniform"
  ],
  "target_rates_hz": [
    1.0,
    10.0
  ],
  "calibration_input_rates": [
    0.001,
    0.01
  ],
  "calibration_steps": 128,
  "evaluation_steps": 256,
  "synthetic_neurons": 2048,
  "synthetic_degree": 48,
  "recurrent_strength": 0.1,
  "local_windows": [
    32,
    64
  ],
  "joint_window": 64,
  "structural_lambdas": [
    0.0,
    0.5
  ],
  "oracle_lambdas": [
    0.5
  ],
  "block_task_sample": 2000,
  "seed": 20260831
}
```

The common simulator is persistent CUDA v1 LIF+PSC (`dt=1 ms`, soft reset). Synthetic graphs use deterministic balanced 80/20 Dale E/I weights. FlyWire preserves signed relative weights. Every graph is normalized to mean absolute recurrent fanout strength 0.1.

## Graph summary

| dataset   |   neurons |   edges |   mean_fanin |   fanin_cv |
|:----------|----------:|--------:|-------------:|-----------:|
| uniform   |      2048 |   97081 |         47.4 |     0.1439 |

## Calibration and selected controls

| dataset   |   control_input_rate |   population_mean_hz |   active_neuron_mean_hz |   silent_fraction |   selected_targets_hz |
|:----------|---------------------:|---------------------:|------------------------:|------------------:|----------------------:|
| uniform   |                0.001 |                 1.06 |                   8.386 |            0.8735 |                     1 |
| uniform   |                0.01  |                 9.85 |                  13.7   |            0.2812 |                    10 |

## Evaluation posterior activity

| dataset   |   control_input_rate |   population_mean_hz |   active_neuron_mean_hz |   silent_fraction |   profile_population_mean_hz |
|:----------|---------------------:|---------------------:|------------------------:|------------------:|-----------------------------:|
| uniform   |                0.001 |                1.019 |                   4.337 |           0.7651  |                        1.053 |
| uniform   |                0.01  |                9.628 |                  10.52  |           0.08447 |                        9.769 |

## Best method at each measured condition

| dataset   |   population_mean_hz | method          |   occupancy_gain |   static_fanout_change |   aggregation_gain |   atomic_reduction_vs_fanout |   atomic_reduction_ci95_low |   atomic_reduction_ci95_high |
|:----------|---------------------:|:----------------|-----------------:|-----------------------:|-------------------:|-----------------------------:|----------------------------:|-----------------------------:|
| uniform   |                1.019 | local_fanin_w32 |        0         |               0        |          0         |                     0        |                    -0.01922 |                      0.01922 |
| uniform   |                9.628 | local_fanin_w64 |       -0.0004561 |              -0.005171 |         -0.0006162 |                     0.006268 |                    -0.01448 |                      0.02702 |

## Best result by method family

| dataset   |   population_mean_hz | method_family   | method            |   occupancy_gain |   atomic_reduction_vs_fanout |   atomic_reduction_ci95_low |
|:----------|---------------------:|:----------------|:------------------|-----------------:|-----------------------------:|----------------------------:|
| uniform   |                1.019 | joint_fanin     | joint_fanin_l0.5  |       -0.005725  |                   -0.0002768 |                    -0.01785 |
| uniform   |                1.019 | local_fanin     | local_fanin_w32   |        0         |                    0         |                    -0.01922 |
| uniform   |                1.019 | oracle_spike    | oracle_spike_l0.5 |       -0.009506  |                   -0.0002372 |                    -0.01707 |
| uniform   |                9.628 | joint_fanin     | joint_fanin_l0    |       -0.0004561 |                    0.003475  |                    -0.01765 |
| uniform   |                9.628 | local_fanin     | local_fanin_w64   |       -0.0004561 |                    0.006268  |                    -0.01448 |
| uniform   |                9.628 | oracle_spike    | oracle_spike_l0.5 |        0.0002282 |                    0.0001088 |                    -0.02127 |

## Oracle profile stability

| dataset   |   population_mean_hz |   profile_nonzero_fraction |   evaluation_nonzero_fraction |   spearman_all |   spearman_any_positive |
|:----------|---------------------:|---------------------------:|------------------------------:|---------------:|------------------------:|
| uniform   |                1.019 |                  0.0003252 |                     0.0002541 |     -0.0002875 |                 -0.8853 |
| uniform   |                9.628 |                  0.02394   |                     0.02302   |     -0.002835  |                 -0.8397 |

## Decision summary

| dataset   |   best_structural_median_atomic_reduction |   best_oracle_median_atomic_reduction | classification                    |
|:----------|------------------------------------------:|--------------------------------------:|:----------------------------------|
| uniform   |                                  0.003134 |                            -6.421e-05 | stop: oracle approximately fanout |

Atomic counts are ratio-of-means estimates from uniformly sampled active BlockTasks; reported intervals propagate independent candidate/baseline standard errors. Occupancy and active-block counts are exact over the entire evaluation trace.
