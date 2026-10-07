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
  "synthetic_recurrent_strength": 1.0,
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
| uniform   |        0.005 |         128 |               0.4607 |             2048 |                        0 |
| uniform   |        0.02  |         128 |               0.8949 |             2048 |                        0 |
| community |        0.005 |         128 |               0.5904 |             2048 |                        0 |
| community |        0.02  |         128 |               0.8981 |             2048 |                        0 |
| spatial   |        0.005 |         128 |               0.6356 |             2048 |                        0 |
| spatial   |        0.02  |         128 |               0.8971 |             2048 |                        0 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                       0.03001  |              0.02478  |                       0.02944 |             0.02955  |
| uniform   |        0.02  |                 0.03275 |                       0.004757 |              0.008586 |                       0.00364 |             0.003539 |
| community |        0.005 |                 0.767   |                       0.2077   |              0.05825  |                       0.1469  |             0.1484   |
| community |        0.02  |                 0.767   |                       0.1058   |              0.003617 |                       0.0612  |             0.06069  |
| spatial   |        0.005 |                 0.9515  |                       0.5397   |              0.1837   |                       0.1408  |             0.1426   |
| spatial   |        0.02  |                 0.9515  |                       0.1648   |              0.03224  |                       0.07414 |             0.07546  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |       -0.001278  |                              0.2979 |                                 0.2847 |         -0.01322   |
| uniform   |        0.02  |       -0.0002547 |                              0.3009 |                                 0.2879 |         -0.013     |
| community |        0.005 |       -0.0009083 |                              0.6681 |                                 0.6695 |          0.001387  |
| community |        0.02  |       -0.0005096 |                              0.6716 |                                 0.6725 |          0.0008307 |
| spatial   |        0.005 |        0.0006826 |                              0.8624 |                                 0.862  |         -0.000397  |
| spatial   |        0.02  |        0.0001276 |                              0.8639 |                                 0.864  |          4.542e-05 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | True                     | False                        | True                           |
| spatial   | True                     | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
