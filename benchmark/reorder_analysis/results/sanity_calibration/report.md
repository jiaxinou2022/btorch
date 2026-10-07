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
  "pair_count": 10000,
  "block_task_sample": 2000,
  "block_size": 32,
  "synthetic_recurrent_strength": 4.0,
  "input_amplitude": 30.0,
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
| uniform   |        0.005 |         256 |               0.9777 |             2048 |                        0 |
| uniform   |        0.02  |         256 |               0.9856 |             2048 |                        0 |
| community |        0.005 |         256 |               0.978  |             2048 |                        0 |
| community |        0.02  |         256 |               0.9862 |             2048 |                        0 |
| spatial   |        0.005 |         256 |               0.978  |             2048 |                        0 |
| spatial   |        0.02  |         256 |               0.9861 |             2048 |                        0 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |               -0.005694 |                       0.007934 |               0.01734 |                      0.003138 |             0.003394 |
| uniform   |        0.02  |               -0.005694 |                       0.0135   |               0.0139  |                      0.01061  |             0.01061  |
| community |        0.005 |                0.7595   |                       0.08674  |               0.02135 |                      0.04855  |             0.04881  |
| community |        0.02  |                0.7595   |                       0.03249  |               0.02147 |                      0.01169  |             0.01175  |
| spatial   |        0.005 |                0.9498   |                       0.3101   |               0.1267  |                      0.08734  |             0.08853  |
| spatial   |        0.02  |                0.9498   |                       0.1031   |               0.04892 |                      0.03685  |             0.03719  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |        0.0005558 |                              0.3028 |                                 0.2897 |         -0.01312   |
| uniform   |        0.02  |       -0.0006124 |                              0.3029 |                                 0.2893 |         -0.01366   |
| community |        0.005 |        0.000247  |                              0.6743 |                                 0.6751 |          0.0007476 |
| community |        0.02  |        0.0003676 |                              0.6744 |                                 0.6752 |          0.000772  |
| spatial   |        0.005 |       -0.0001235 |                              0.8667 |                                 0.8665 |         -0.0002413 |
| spatial   |        0.02  |       -6.121e-05 |                              0.8665 |                                 0.8665 |         -8.845e-05 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | True                           |
| spatial   | True                     | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
