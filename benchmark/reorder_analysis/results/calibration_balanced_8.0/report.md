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
  "synthetic_recurrent_strength": 8.0,
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
| uniform   |        0.005 |         256 |               0.4192 |             1621 |                   0.2085 |
| uniform   |        0.02  |         256 |               0.4215 |             1596 |                   0.2207 |
| community |        0.005 |         256 |               0.4321 |             1573 |                   0.2319 |
| community |        0.02  |         256 |               0.4321 |             1617 |                   0.2104 |
| spatial   |        0.005 |         256 |               0.3834 |             1527 |                   0.2544 |
| spatial   |        0.02  |         256 |               0.3671 |             1593 |                   0.2222 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                        0.03901 |              0.01607  |                      0.006999 |            -0.01514  |
| uniform   |        0.02  |                 0.03275 |                        0.02943 |              0.004053 |                      0.006348 |            -0.001883 |
| community |        0.005 |                 0.767   |                        0.04328 |              0.09128  |                      0.03607  |             0.003161 |
| community |        0.02  |                 0.767   |                        0.02886 |              0.05863  |                      0.007633 |            -0.02105  |
| spatial   |        0.005 |                 0.9515  |                        0.1335  |              0.2104   |                      0.06488  |             0.01611  |
| spatial   |        0.02  |                 0.9515  |                        0.1359  |              0.2289   |                      0.07826  |             0.02509  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |        0.0002483 |                              0.1515 |                                 0.1419 |          -0.009616 |
| uniform   |        0.02  |       -0.0005527 |                              0.1505 |                                 0.1409 |          -0.009525 |
| community |        0.005 |        0.0005596 |                              0.5013 |                                 0.4979 |          -0.003412 |
| community |        0.02  |       -0.0002457 |                              0.4943 |                                 0.4958 |           0.001501 |
| spatial   |        0.005 |       -0.008073  |                              0.7525 |                                 0.7546 |           0.002082 |
| spatial   |        0.02  |       -0.001295  |                              0.7415 |                                 0.745  |           0.003553 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | False                          |
| spatial   | True                     | False                        | True                           |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
