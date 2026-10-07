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
| uniform   |        0.005 |         256 |               0.3033 |             1580 |                   0.2285 |
| uniform   |        0.02  |         256 |               0.3388 |             1615 |                   0.2114 |
| community |        0.005 |         256 |               0.2897 |             1512 |                   0.2617 |
| community |        0.02  |         256 |               0.344  |             1566 |                   0.2354 |
| spatial   |        0.005 |         256 |               0.2451 |             1465 |                   0.2847 |
| spatial   |        0.02  |         256 |               0.2797 |             1531 |                   0.2524 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                        0.02661 |               0.02227 |                     -0.00127  |              0.03878 |
| uniform   |        0.02  |                 0.03275 |                        0.04764 |               0.04879 |                      0.01719  |              0.03502 |
| community |        0.005 |                 0.767   |                        0.02808 |               0.03644 |                      0.000585 |              0.01201 |
| community |        0.02  |                 0.767   |                        0.03034 |               0.04197 |                      0.01242  |              0.01329 |
| spatial   |        0.005 |                 0.9515  |                        0.1367  |               0.1722  |                      0.08015  |              0.0565  |
| spatial   |        0.02  |                 0.9515  |                        0.1461  |               0.2397  |                      0.08457  |              0.05554 |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |       -0.0003447 |                              0.1286 |                                 0.1172 |         -0.01138   |
| uniform   |        0.02  |       -0.0008136 |                              0.1285 |                                 0.1188 |         -0.009712  |
| community |        0.005 |        0.0005682 |                              0.4462 |                                 0.4469 |          0.0006674 |
| community |        0.02  |        0.00119   |                              0.4602 |                                 0.4577 |         -0.002518  |
| spatial   |        0.005 |        0.005937  |                              0.7202 |                                 0.7183 |         -0.001859  |
| spatial   |        0.02  |        0.002775  |                              0.7245 |                                 0.7235 |         -0.001016  |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | False                          |
| spatial   | True                     | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
