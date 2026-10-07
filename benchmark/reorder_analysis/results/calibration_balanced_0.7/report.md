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
  "synthetic_recurrent_strength": 0.7,
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
| uniform   |        0.005 |         512 |             0.003638 |             1698 |                 0.1709   |
| uniform   |        0.02  |         512 |             0.01482  |             2037 |                 0.005371 |
| community |        0.005 |         512 |             0.003661 |             1694 |                 0.1729   |
| community |        0.02  |         512 |             0.01572  |             2035 |                 0.006348 |
| spatial   |        0.005 |         512 |             0.003542 |             1660 |                 0.1895   |
| spatial   |        0.02  |         512 |             0.07301  |             1936 |                 0.05469  |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                      -0.04787  |             -0.05767  |                     -0.03942  |            -0.0464   |
| uniform   |        0.02  |                 0.03275 |                       0.008851 |              0.008927 |                      0.01075  |             0.01744  |
| community |        0.005 |                 0.767   |                      -0.003337 |             -0.003502 |                     -0.008513 |            -0.007282 |
| community |        0.02  |                 0.767   |                       0.01475  |              0.01532  |                     -0.01506  |            -0.02041  |
| spatial   |        0.005 |                 0.9515  |                       0.0266   |              0.03182  |                      0.002526 |             0.009511 |
| spatial   |        0.02  |                 0.9515  |                       0.05572  |              0.03768  |                      0.07241  |             0.03029  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |       -0.01075   |                            0.0015   |                               0.001335 |         -0.0001647 |
| uniform   |        0.02  |       -0.002414  |                            0.005461 |                               0.005423 |         -3.87e-05  |
| community |        0.005 |       -0.002485  |                            0.007665 |                               0.006947 |         -0.0007181 |
| community |        0.02  |        0.003506  |                            0.03132  |                               0.03263  |          0.001307  |
| spatial   |        0.005 |       -0.0008589 |                            0.02356  |                               0.02483  |          0.001263  |
| spatial   |        0.02  |       -5.715e-05 |                            0.5822   |                               0.5867   |          0.004517  |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | True                           |
| spatial   | False                    | False                        | True                           |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
