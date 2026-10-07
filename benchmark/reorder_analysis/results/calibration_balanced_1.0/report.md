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
| uniform   |        0.005 |         256 |             0.003492 |             1176 |                   0.4258 |
| uniform   |        0.02  |         256 |             0.03619  |             1777 |                   0.1323 |
| community |        0.005 |         256 |             0.003531 |             1180 |                   0.4238 |
| community |        0.02  |         256 |             0.1393   |             1683 |                   0.1782 |
| spatial   |        0.005 |         256 |             0.003635 |             1186 |                   0.4209 |
| spatial   |        0.02  |         256 |             0.1462   |             1702 |                   0.1689 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.005 |                 0.03275 |                      -0.004014 |             -0.004439 |                     -0.002857 |            -0.007456 |
| uniform   |        0.02  |                 0.03275 |                       0.02027  |              0.01716  |                      0.008571 |            -0.01949  |
| community |        0.005 |                 0.767   |                       0.009412 |              0.01755  |                     -0.004856 |            -0.003241 |
| community |        0.02  |                 0.767   |                       0.06041  |              0.09133  |                      0.05539  |             0.01557  |
| spatial   |        0.005 |                 0.9515  |                      -0.005187 |             -0.009065 |                     -0.01327  |            -0.03305  |
| spatial   |        0.02  |                 0.9515  |                       0.09339  |              0.101    |                      0.08311  |             0.04085  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.005 |       -0.01549   |                            0.001738 |                               0.001174 |         -0.0005639 |
| uniform   |        0.02  |        0.003514  |                            0.01561  |                               0.01475  |         -0.0008608 |
| community |        0.005 |        0.002303  |                            0.008109 |                               0.008229 |          0.0001207 |
| community |        0.02  |        0.007487  |                            0.3293   |                               0.34     |          0.01066   |
| spatial   |        0.005 |       -0.001694  |                            0.03057  |                               0.02931  |         -0.001253  |
| spatial   |        0.02  |        0.0002652 |                            0.6655   |                               0.6646   |         -0.0008394 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | True                           |
| spatial   | False                    | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
