# Fanout-preserving refinement experiment (v2)

Numerical results only; no images or visualization files were generated.
Profile and evaluation traces use different seeds. Oracle orders are built exclusively from the profile trace.

## Configuration

```json
{
  "datasets": [
    "flybrain"
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

| dataset   |   neurons |    edges |   mean_fanin |   fanin_cv |
|:----------|----------:|---------:|-------------:|-----------:|
| flybrain  |    138639 | 15091983 |        108.9 |      1.482 |

## Calibration and selected controls

| dataset   |   control_input_rate |   population_mean_hz |   active_neuron_mean_hz |   silent_fraction |   selected_targets_hz |
|:----------|---------------------:|---------------------:|------------------------:|------------------:|----------------------:|
| flybrain  |                0.001 |                1.006 |                   8.305 |            0.8789 |                     1 |
| flybrain  |                0.01  |               11.67  |                  17     |            0.3138 |                    10 |

## Evaluation posterior activity

| dataset   |   control_input_rate |   population_mean_hz |   active_neuron_mean_hz |   silent_fraction |   profile_population_mean_hz |
|:----------|---------------------:|---------------------:|------------------------:|------------------:|-----------------------------:|
| flybrain  |                0.001 |                1.004 |                   4.428 |            0.7732 |                        1.006 |
| flybrain  |                0.01  |               12.29  |                  13.94  |            0.118  |                       12.36  |

## Best method at each measured condition

| dataset   |   population_mean_hz | method            |   occupancy_gain |   static_fanout_change |   aggregation_gain |   atomic_reduction_vs_fanout |   atomic_reduction_ci95_low |   atomic_reduction_ci95_high |
|:----------|---------------------:|:------------------|-----------------:|-----------------------:|-------------------:|-----------------------------:|----------------------------:|-----------------------------:|
| flybrain  |                1.004 | joint_fanin_l0.5  |        -0.001196 |                0.03725 |          0.0009898 |                      0.04967 |                    -0.1093  |                       0.2086 |
| flybrain  |               12.29  | oracle_spike_l0.5 |         0.007827 |                0.03726 |         -0.006057  |                      0.04497 |                    -0.09531 |                       0.1852 |

## Best result by method family

| dataset   |   population_mean_hz | method_family   | method            |   occupancy_gain |   atomic_reduction_vs_fanout |   atomic_reduction_ci95_low |
|:----------|---------------------:|:----------------|:------------------|-----------------:|-----------------------------:|----------------------------:|
| flybrain  |                1.004 | joint_fanin     | joint_fanin_l0.5  |        -0.001196 |                     0.04967  |                    -0.1093  |
| flybrain  |                1.004 | local_fanin     | local_fanin_w32   |         0        |                     0        |                    -0.1579  |
| flybrain  |                1.004 | oracle_spike    | oracle_spike_l0.5 |        -0.001253 |                     0.002961 |                    -0.1538  |
| flybrain  |               12.29  | joint_fanin     | joint_fanin_l0    |         0.007658 |                    -0.004543 |                    -0.1477  |
| flybrain  |               12.29  | local_fanin     | local_fanin_w32   |         0        |                     0        |                    -0.1441  |
| flybrain  |               12.29  | oracle_spike    | oracle_spike_l0.5 |         0.007827 |                     0.04497  |                    -0.09531 |

## Oracle profile stability

| dataset   |   population_mean_hz |   profile_nonzero_fraction |   evaluation_nonzero_fraction |   persistence_given_profile |   positive_support_jaccard |   spearman_all |   spearman_any_positive |
|:----------|---------------------:|---------------------------:|------------------------------:|----------------------------:|---------------------------:|---------------:|------------------------:|
| flybrain  |                1.004 |                  0.0002542 |                     0.0003152 |                      0      |                     0      |     -0.0002831 |                 -0.8856 |
| flybrain  |               12.29  |                  0.03284   |                     0.03173   |                      0.1918 |                     0.1081 |      0.1618    |                 -0.8027 |

## Decision summary

| dataset   |   best_structural_median_atomic_reduction |   best_oracle_median_atomic_reduction | classification                     |
|:----------|------------------------------------------:|--------------------------------------:|:-----------------------------------|
| flybrain  |                                   0.02483 |                               0.02396 | structural refinement has headroom |

Atomic counts are ratio-of-means estimates from uniformly sampled active BlockTasks; reported intervals propagate independent candidate/baseline standard errors. Occupancy and active-block counts are exact over the entire evaluation trace.
