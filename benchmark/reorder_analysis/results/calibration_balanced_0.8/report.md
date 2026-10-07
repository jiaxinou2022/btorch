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
  "synthetic_timesteps": 512,
  "flybrain_timesteps": 128,
  "pair_count": 2000,
  "block_task_sample": 2000,
  "block_size": 32,
  "synthetic_recurrent_strength": 0.8,
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
| uniform   |        0.005 |         512 |             0.003565 |             1686 |                  0.1768  |
| uniform   |        0.02  |         512 |             0.01647  |             2021 |                  0.01318 |
| community |        0.005 |         512 |             0.003576 |             1687 |                  0.1763  |
| community |        0.02  |         512 |             0.02198  |             1997 |                  0.0249  |
| spatial   |        0.005 |         512 |             0.003504 |             1658 |                  0.1904  |
| spatial   |        0.02  |         512 |             0.1045   |             1845 |                  0.09912 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                      -0.04472  |              -0.05463 |                     -0.03107  |            -0.0356   |
| uniform   |        0.02  |                 0.03275 |                       0.009881 |               0.01113 |                      0.02152  |             0.0367   |
| community |        0.005 |                 0.767   |                      -0.0231   |              -0.02732 |                     -0.0383   |            -0.04725  |
| community |        0.02  |                 0.767   |                       0.02478  |               0.01829 |                     -0.01138  |            -0.02943  |
| spatial   |        0.005 |                 0.9515  |                       0.02662  |               0.03147 |                      0.001817 |             0.007913 |
| spatial   |        0.02  |                 0.9515  |                       0.06815  |               0.05592 |                      0.08471  |             0.02216  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |       -0.01155   |                            0.001527 |                               0.00122  |         -0.0003064 |
| uniform   |        0.02  |       -0.005477  |                            0.007113 |                               0.00531  |         -0.001802  |
| community |        0.005 |       -0.002264  |                            0.007278 |                               0.007709 |          0.0004311 |
| community |        0.02  |       -0.0001298 |                            0.05863  |                               0.05668  |         -0.001948  |
| spatial   |        0.005 |       -0.002601  |                            0.02284  |                               0.02226  |         -0.0005823 |
| spatial   |        0.02  |        0.00148   |                            0.6147   |                               0.6165   |          0.001759  |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | False                          |
| spatial   | False                    | False                        | True                           |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
