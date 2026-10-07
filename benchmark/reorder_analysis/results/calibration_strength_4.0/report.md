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
| uniform   |        0.005 |         128 |               0.9254 |             2048 |                        0 |
| uniform   |        0.02  |         128 |               0.9618 |             2048 |                        0 |
| community |        0.005 |         128 |               0.9277 |             2048 |                        0 |
| community |        0.02  |         128 |               0.9633 |             2048 |                        0 |
| spatial   |        0.005 |         128 |               0.9311 |             2048 |                        0 |
| spatial   |        0.02  |         128 |               0.9632 |             2048 |                        0 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                       -0.01334 |             -0.01147  |                      -0.02467 |             -0.02437 |
| uniform   |        0.02  |                 0.03275 |                        0.02304 |             -0.006755 |                       0.03153 |              0.03141 |
| community |        0.005 |                 0.767   |                        0.1903  |              0.0536   |                       0.06276 |              0.06263 |
| community |        0.02  |                 0.767   |                        0.07039 |             -0.0219   |                       0.0531  |              0.05347 |
| spatial   |        0.005 |                 0.9515  |                        0.4592  |              0.169    |                       0.1226  |              0.1229  |
| spatial   |        0.02  |                 0.9515  |                        0.1617  |              0.06742  |                       0.07625 |              0.07671 |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |        0.0007709 |                              0.3026 |                                 0.2891 |         -0.01355   |
| uniform   |        0.02  |       -0.0009902 |                              0.3026 |                                 0.289  |         -0.01357   |
| community |        0.005 |        0.000899  |                              0.6735 |                                 0.6746 |          0.001169  |
| community |        0.02  |       -0.0002477 |                              0.674  |                                 0.6748 |          0.0008734 |
| spatial   |        0.005 |        0.0003847 |                              0.8661 |                                 0.8659 |         -0.0001749 |
| spatial   |        0.02  |       -0.0001237 |                              0.8658 |                                 0.8658 |         -6.846e-06 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | True                     | False                        | True                           |
| spatial   | True                     | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
