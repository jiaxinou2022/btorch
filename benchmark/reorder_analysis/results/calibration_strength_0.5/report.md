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
  "synthetic_recurrent_strength": 0.5,
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
| uniform   |        0.005 |         128 |             0.00489  |              955 |                   0.5337 |
| uniform   |        0.02  |         128 |             0.7714   |             2048 |                   0      |
| community |        0.005 |         128 |             0.004906 |              940 |                   0.541  |
| community |        0.02  |         128 |             0.7745   |             2048 |                   0      |
| spatial   |        0.005 |         128 |             0.005199 |              984 |                   0.5195 |
| spatial   |        0.02  |         128 |             0.7879   |             2048 |                   0      |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                      -0.01653  |             -0.02875  |                      -0.01294 |             -0.02092 |
| uniform   |        0.02  |                 0.03275 |                      -0.02902  |             -0.02246  |                      -0.01844 |             -0.01847 |
| community |        0.005 |                 0.767   |                      -0.02014  |             -0.03403  |                      -0.03036 |             -0.05354 |
| community |        0.02  |                 0.767   |                       0.03305  |             -0.009211 |                       0.03277 |              0.03123 |
| spatial   |        0.005 |                 0.9515  |                      -0.003972 |             -0.001431 |                      -0.02196 |             -0.06018 |
| spatial   |        0.02  |                 0.9515  |                       0.1931   |              0.06114  |                       0.02781 |              0.02898 |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |        -0.01343  |                            0.002267 |                               0.001643 |         -0.0006242 |
| uniform   |        0.02  |        -0.002959 |                            0.2976   |                               0.2843   |         -0.0133    |
| community |        0.005 |         0.002536 |                            0.01136  |                               0.0112   |         -0.0001565 |
| community |        0.02  |         0.001481 |                            0.6665   |                               0.6666   |          2.858e-05 |
| spatial   |        0.005 |         0.005516 |                            0.02485  |                               0.02593  |          0.001079  |
| spatial   |        0.02  |         0        |                            0.8591   |                               0.8595   |          0.0004579 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | False                          |
| spatial   | False                    | False                        | True                           |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
