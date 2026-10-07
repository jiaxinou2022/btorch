# Fanin / co-spiking preprocessing experiment

This report contains numerical analysis only; no figures were generated.
Pair sampling is 50% uniform random and 50% conditioned on sharing at least one fanin. Binned trends are therefore conditional comparisons, not estimates of the population frequency of similarity bins.
Destination-union metrics use a reproducible uniform sample of active BlockTasks; occupancy and active-block counts are exact over every step.

## Configuration

```json
{
  "datasets": [
    "uniform",
    "community",
    "spatial"
  ],
  "input_rates": [
    0.005,
    0.02
  ],
  "synthetic_neurons": 2048,
  "synthetic_degree": 48,
  "synthetic_timesteps": 256,
  "flybrain_timesteps": 128,
  "pair_count": 2000,
  "block_task_sample": 2000,
  "block_size": 32,
  "synthetic_recurrent_strength": 0.2,
  "input_amplitude": 1.1,
  "seed": 20260830
}
```

## Graphs

| dataset   |   neurons |   edges |   density |   mean_fanin |   fanin_cv |   max_fanin |   max_fanout |
|:----------|----------:|--------:|----------:|-------------:|-----------:|------------:|-------------:|
| uniform   |      2048 |   97239 |   0.02318 |        47.48 |     0.1406 |          75 |           48 |
| community |      2048 |   91713 |   0.02187 |        44.78 |     0.138  |          66 |           48 |
| spatial   |      2048 |   71104 |   0.01695 |        34.72 |     0.1258 |          50 |           42 |

## Measured activity

| dataset   |   input_rate |   timesteps |   actual_firing_rate |   active_neurons |   silent_neuron_fraction |
|:----------|-------------:|------------:|---------------------:|-----------------:|-------------------------:|
| uniform   |        0.005 |         256 |             0.004902 |             1455 |                 0.2896   |
| uniform   |        0.02  |         256 |             0.0195   |             2035 |                 0.006348 |
| community |        0.005 |         256 |             0.005005 |             1482 |                 0.2764   |
| community |        0.02  |         256 |             0.02027  |             2043 |                 0.002441 |
| spatial   |        0.005 |         256 |             0.005095 |             1497 |                 0.269    |
| spatial   |        0.02  |         256 |             0.02045  |             2039 |                 0.004395 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                       0.002297 |              0.004783 |                      0.007043 |             0.01254  |
| uniform   |        0.02  |                 0.03275 |                       0.04094  |              0.04073  |                      0.02545  |             0.008608 |
| community |        0.005 |                 0.767   |                       0.002313 |              0.002365 |                     -0.01252  |            -0.01541  |
| community |        0.02  |                 0.767   |                      -0.02653  |             -0.02564  |                     -0.02632  |            -0.02467  |
| spatial   |        0.005 |                 0.9515  |                      -0.01454  |             -0.01684  |                     -0.01579  |            -0.02679  |
| spatial   |        0.02  |                 0.9515  |                       0.008524 |              0.009464 |                      0.02089  |             0.02208  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |       -0.01457   |                            0.001979 |                               0.001409 |         -0.0005707 |
| uniform   |        0.02  |       -0.003513  |                            0.007601 |                               0.007056 |         -0.0005451 |
| community |        0.005 |        0.0004137 |                            0.0104   |                               0.01034  |         -6.128e-05 |
| community |        0.02  |        0.0124    |                            0.03876  |                               0.04066  |          0.0019    |
| spatial   |        0.005 |        0.004418  |                            0.02441  |                               0.02571  |          0.001296  |
| spatial   |        0.02  |       -0.001508  |                            0.1139   |                               0.1108   |         -0.003082  |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | True                           |
| spatial   | False                    | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
