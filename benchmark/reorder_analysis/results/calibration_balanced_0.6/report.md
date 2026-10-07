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
  "synthetic_recurrent_strength": 0.6,
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
| uniform   |        0.005 |         512 |             0.003743 |             1725 |                 0.1577   |
| uniform   |        0.02  |         512 |             0.01444  |             2038 |                 0.004883 |
| community |        0.005 |         512 |             0.00376  |             1721 |                 0.1597   |
| community |        0.02  |         512 |             0.01517  |             2038 |                 0.004883 |
| spatial   |        0.005 |         512 |             0.003632 |             1683 |                 0.1782   |
| spatial   |        0.02  |         512 |             0.05916  |             1971 |                 0.0376   |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                      -0.03268  |             -0.03937  |                     -0.02794  |            -0.02999  |
| uniform   |        0.02  |                 0.03275 |                       0.0125   |              0.01248  |                      0.02183  |             0.03316  |
| community |        0.005 |                 0.767   |                      -0.02463  |             -0.02973  |                     -0.02909  |            -0.03455  |
| community |        0.02  |                 0.767   |                      -0.006301 |             -0.006529 |                     -0.03933  |            -0.02774  |
| spatial   |        0.005 |                 0.9515  |                       0.02727  |              0.03277  |                     -0.007216 |            -0.002787 |
| spatial   |        0.02  |                 0.9515  |                       0.04524  |              0.02728  |                      0.07541  |             0.02707  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |        -0.01126  |                            0.001564 |                               0.00128  |         -0.0002839 |
| uniform   |        0.02  |        -0.00312  |                            0.006308 |                               0.005067 |         -0.001241  |
| community |        0.005 |        -0.003231 |                            0.007947 |                               0.007798 |         -0.0001489 |
| community |        0.02  |         0.005725 |                            0.03024  |                               0.0288   |         -0.001441  |
| spatial   |        0.005 |         0        |                            0.02373  |                               0.02396  |          0.0002333 |
| spatial   |        0.02  |        -0.001324 |                            0.5737   |                               0.5654   |         -0.008297  |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | False                          |
| spatial   | False                    | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
