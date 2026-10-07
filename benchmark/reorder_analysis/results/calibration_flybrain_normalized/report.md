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
  "flybrain_timesteps": 256,
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
| flybrain  |        0.001 |         256 |              0.04672 |            36390 |                  0.7375  |
| flybrain  |        0.005 |         256 |              0.04989 |            68054 |                  0.5091  |
| flybrain  |        0.01  |         256 |              0.05449 |            85888 |                  0.3805  |
| flybrain  |        0.02  |         256 |              0.06133 |           102545 |                  0.2603  |
| flybrain  |        0.05  |         256 |              0.08182 |           120156 |                  0.1333  |
| flybrain  |        0.1   |         256 |              0.1172  |           129191 |                  0.06815 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| flybrain  |        0.001 |                  0.8523 |                       0.1403   |             0.265     |                     -0.003712 |             0.03923  |
| flybrain  |        0.005 |                  0.8523 |                       0.04139  |             0.04502   |                     -0.00805  |            -0.02449  |
| flybrain  |        0.01  |                  0.8523 |                       0.05384  |             0.07058   |                     -0.009471 |            -0.0303   |
| flybrain  |        0.02  |                  0.8523 |                       0.04194  |             0.04988   |                     -0.01161  |            -0.002322 |
| flybrain  |        0.05  |                  0.8523 |                      -0.004052 |             0.0008344 |                     -0.03494  |            -0.009536 |
| flybrain  |        0.1   |                  0.8523 |                       0.02576  |             0.0367    |                     -0.0169   |             0.001055 |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| flybrain  |        0.001 |         0.05682  |                              0.159  |                                 0.1726 |           0.01353  |
| flybrain  |        0.005 |         0.0508   |                              0.1556 |                                 0.1652 |           0.009608 |
| flybrain  |        0.01  |         0.04298  |                              0.1516 |                                 0.1569 |           0.005257 |
| flybrain  |        0.02  |         0.03256  |                              0.159  |                                 0.1508 |          -0.008146 |
| flybrain  |        0.05  |         0.01523  |                              0.1343 |                                 0.1389 |           0.004589 |
| flybrain  |        0.1   |         0.005473 |                              0.1223 |                                 0.1291 |           0.006784 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| flybrain  | False                    | False                        | True                           |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
