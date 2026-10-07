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
  "synthetic_recurrent_strength": 4.0,
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
| uniform   |        0.005 |         256 |               0.3752 |             1566 |                   0.2354 |
| uniform   |        0.02  |         256 |               0.3958 |             1550 |                   0.2432 |
| community |        0.005 |         256 |               0.4109 |             1546 |                   0.2451 |
| community |        0.02  |         256 |               0.4151 |             1588 |                   0.2246 |
| spatial   |        0.005 |         256 |               0.3419 |             1463 |                   0.2856 |
| spatial   |        0.02  |         256 |               0.3516 |             1584 |                   0.2266 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                        0.01883 |              0.02789  |                      0.02156  |              0.04112 |
| uniform   |        0.02  |                 0.03275 |                        0.03585 |              0.004878 |                     -0.007981 |             -0.04686 |
| community |        0.005 |                 0.767   |                        0.03358 |             -0.003451 |                      0.02289  |              0.01107 |
| community |        0.02  |                 0.767   |                        0.03414 |              0.03663  |                      0.01498  |              0.01074 |
| spatial   |        0.005 |                 0.9515  |                        0.1327  |              0.2182   |                      0.09403  |              0.03011 |
| spatial   |        0.02  |                 0.9515  |                        0.1366  |              0.2408   |                      0.09228  |              0.07339 |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |        6.31e-05  |                              0.1386 |                                 0.1308 |         -0.007798  |
| uniform   |        0.02  |        0.0003089 |                              0.1431 |                                 0.1343 |         -0.008869  |
| community |        0.005 |       -0.001964  |                              0.4979 |                                 0.496  |         -0.001816  |
| community |        0.02  |        6.173e-05 |                              0.4907 |                                 0.4903 |         -0.0003636 |
| spatial   |        0.005 |        0.001914  |                              0.7424 |                                 0.7383 |         -0.004111  |
| spatial   |        0.02  |        0.002021  |                              0.7395 |                                 0.7411 |          0.001593  |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | False                          |
| spatial   | True                     | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
