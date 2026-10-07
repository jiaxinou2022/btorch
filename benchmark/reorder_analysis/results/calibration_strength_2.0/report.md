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
  "synthetic_timesteps": 128,
  "flybrain_timesteps": 128,
  "pair_count": 2000,
  "block_task_sample": 2000,
  "block_size": 32,
  "synthetic_recurrent_strength": 2.0,
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
| uniform   |        0.005 |         128 |               0.8483 |             2048 |                        0 |
| uniform   |        0.02  |         128 |               0.9391 |             2048 |                        0 |
| community |        0.005 |         128 |               0.8507 |             2048 |                        0 |
| community |        0.02  |         128 |               0.9424 |             2048 |                        0 |
| spatial   |        0.005 |         128 |               0.879  |             2048 |                        0 |
| spatial   |        0.02  |         128 |               0.9411 |             2048 |                        0 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                      -0.001193 |              0.002786 |                     -0.009416 |            -0.009297 |
| uniform   |        0.02  |                 0.03275 |                      -0.004563 |             -0.008086 |                     -0.01318  |            -0.01365  |
| community |        0.005 |                 0.767   |                       0.2231   |              0.0155   |                      0.1204   |             0.1218   |
| community |        0.02  |                 0.767   |                       0.1071   |              0.001747 |                      0.07731  |             0.07721  |
| spatial   |        0.005 |                 0.9515  |                       0.5659   |              0.1964   |                      0.145    |             0.1475   |
| spatial   |        0.02  |                 0.9515  |                       0.2162   |              0.0853   |                      0.07117  |             0.07111  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |       -0.001098  |                              0.302  |                                 0.2883 |         -0.01365   |
| uniform   |        0.02  |       -0.001749  |                              0.3016 |                                 0.2887 |         -0.01296   |
| community |        0.005 |       -0.0004116 |                              0.6727 |                                 0.6738 |          0.001052  |
| community |        0.02  |        0.0001249 |                              0.6732 |                                 0.6742 |          0.0009722 |
| spatial   |        0.005 |       -0.0001336 |                              0.8656 |                                 0.8652 |         -0.0003704 |
| spatial   |        0.02  |        0.0004998 |                              0.8657 |                                 0.8653 |         -0.0004025 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | True                     | False                        | True                           |
| spatial   | True                     | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
