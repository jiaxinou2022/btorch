# Fanin / co-spiking preprocessing experiment

This report contains numerical analysis only; no figures were generated.
Pair sampling is 50% uniform random and 50% conditioned on sharing at least one fanin. Binned trends are therefore conditional comparisons, not estimates of the population frequency of similarity bins.
Destination-union metrics use a reproducible uniform sample of active BlockTasks; occupancy and active-block counts are exact over every step.

## Configuration

```json
{
  "datasets": [
    "flybrain"
  ],
  "input_rates": [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1
  ],
  "synthetic_neurons": 8192,
  "synthetic_degree": 64,
  "synthetic_timesteps": 2048,
  "flybrain_timesteps": 128,
  "pair_count": 2000,
  "block_task_sample": 10000,
  "block_size": 32,
  "synthetic_recurrent_strength": 0.6,
  "input_amplitude": 1.1,
  "seed": 20260830
}
```

## Graphs

| dataset   |   neurons |    edges |   density |   mean_fanin |   fanin_cv |   max_fanin |   max_fanout |
|:----------|----------:|---------:|----------:|-------------:|-----------:|------------:|-------------:|
| flybrain  |    138639 | 15091983 | 0.0007852 |        108.9 |      1.482 |       10356 |         9783 |

## Measured activity

| dataset   |   input_rate |   timesteps |   actual_firing_rate |   active_neurons |   silent_neuron_fraction |
|:----------|-------------:|------------:|---------------------:|-----------------:|-------------------------:|
| flybrain  |        0.001 |         128 |               0.2596 |            77924 |                   0.4379 |
| flybrain  |        0.005 |         128 |               0.2536 |            73871 |                   0.4672 |
| flybrain  |        0.01  |         128 |               0.2622 |            77346 |                   0.4421 |
| flybrain  |        0.02  |         128 |               0.2644 |            84446 |                   0.3909 |
| flybrain  |        0.05  |         128 |               0.2792 |            93020 |                   0.329  |
| flybrain  |        0.1   |         128 |               0.2762 |            97726 |                   0.2951 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| flybrain  |        0.001 |                  0.8523 |                         0.1541 |                0.1632 |                        0.162  |              0.03372 |
| flybrain  |        0.005 |                  0.8523 |                         0.1848 |                0.2505 |                        0.2269 |              0.03927 |
| flybrain  |        0.01  |                  0.8523 |                         0.1735 |                0.2171 |                        0.1688 |              0.05631 |
| flybrain  |        0.02  |                  0.8523 |                         0.2    |                0.2062 |                        0.1748 |              0.07704 |
| flybrain  |        0.05  |                  0.8523 |                         0.2038 |                0.2516 |                        0.1377 |              0.03543 |
| flybrain  |        0.1   |                  0.8523 |                         0.1966 |                0.2142 |                        0.1542 |              0.0679  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| flybrain  |        0.001 |         0.01111  |                              0.1327 |                                 0.1332 |          0.0005056 |
| flybrain  |        0.005 |         0.0104   |                              0.1323 |                                 0.1312 |         -0.001169  |
| flybrain  |        0.01  |         0.008792 |                              0.1316 |                                 0.1294 |         -0.00221   |
| flybrain  |        0.02  |         0.006948 |                              0.1326 |                                 0.1374 |          0.004819  |
| flybrain  |        0.05  |         0.002716 |                              0.1341 |                                 0.1356 |          0.001478  |
| flybrain  |        0.1   |         0.00157  |                              0.1328 |                                 0.1333 |          0.0004976 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| flybrain  | True                     | False                        | True                           |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
