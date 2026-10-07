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
  "pair_count": 1000,
  "block_task_sample": 10000,
  "block_size": 32,
  "synthetic_recurrent_strength": 0.2,
  "input_amplitude": 1.1,
  "seed": 20260830
}
```

The simulator is the repository's persistent CUDA v1 LIF+PSC path (`dt=1`, threshold 1, soft reset). External event amplitude is 1.1, so one event leaves only a 0.1 post-reset residual. Synthetic graphs use a deterministic 80/20 Dale E/I split with inhibitory edges 4x stronger and mean recurrent strength 0.6. FlyWire retains signed relative synapse counts but is normalized to the same mean absolute fanout strength; this is a common topology workload, not a faithful Shiu FlyBrain reproduction.

## Graphs

| dataset   |   neurons |    edges |   density |   mean_fanin |   fanin_cv |   max_fanin |   max_fanout |
|:----------|----------:|---------:|----------:|-------------:|-----------:|------------:|-------------:|
| flybrain  |    138639 | 15091983 | 0.0007852 |        108.9 |      1.482 |       10356 |         9783 |

## Measured activity

| dataset   |   input_rate |   timesteps |   actual_firing_rate |   active_neurons |   silent_neuron_fraction |
|:----------|-------------:|------------:|---------------------:|-----------------:|-------------------------:|
| flybrain  |        0.001 |         128 |            0.0009819 |            16315 |                  0.8823  |
| flybrain  |        0.005 |         128 |            0.01116   |            61678 |                  0.5551  |
| flybrain  |        0.01  |         128 |            0.0165    |            88385 |                  0.3625  |
| flybrain  |        0.02  |         128 |            0.02535   |           112152 |                  0.1911  |
| flybrain  |        0.05  |         128 |            0.05162   |           132979 |                  0.04083 |
| flybrain  |        0.1   |         128 |            0.09784   |           137033 |                  0.01158 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| flybrain  |        0.001 |                  0.8596 |                       0.002243 |            -0.1404    |                     0.003039  |           nan        |
| flybrain  |        0.005 |                  0.8596 |                       0.1158   |             0.2076    |                    -0.0007316 |            -0.08077  |
| flybrain  |        0.01  |                  0.8596 |                       0.09954  |             0.1362    |                     0.02307   |             0.04289  |
| flybrain  |        0.02  |                  0.8596 |                       0.1185   |             0.1456    |                     0.03399   |             0.04151  |
| flybrain  |        0.05  |                  0.8596 |                       0.01847  |             0.01534   |                     0.009618  |             0.01286  |
| flybrain  |        0.1   |                  0.8596 |                       0.02824  |            -0.0002174 |                     0.01115   |             0.009238 |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |   aggregation_gain_ci95_low |   aggregation_gain_ci95_high |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|----------------------------:|-----------------------------:|
| flybrain  |        0.001 |        -0.000233 |                            0.001463 |                               0.000246 |         -0.001217  |                  -0.003688  |                     0.001253 |
| flybrain  |        0.005 |         0.04425  |                            0.1099   |                               0.1197   |          0.009799  |                  -0.00616   |                     0.02576  |
| flybrain  |        0.01  |         0.03185  |                            0.1039   |                               0.1034   |         -0.0004455 |                  -0.01565   |                     0.01476  |
| flybrain  |        0.02  |         0.01955  |                            0.07877  |                               0.09466  |          0.01589   |                   0.0002633 |                     0.03152  |
| flybrain  |        0.05  |         0.007483 |                            0.07326  |                               0.07712  |          0.003859  |                  -0.01032   |                     0.01803  |
| flybrain  |        0.1   |         0.001116 |                            0.06564  |                               0.06585  |          0.0002076 |                  -0.008161  |                     0.008576 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_gain_lower_CI_positive   |
|:----------|:-------------------------|:-----------------------------|:----------------------------|
| flybrain  | False                    | False                        | False                       |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.

## Data-first interpretation

| dataset   |   median_raw_fanin_spike_corr |   median_partial_fanin_spike_corr |   median_occupancy_gain_pct |   max_occupancy_gain_pct |   aggregation_positive_CI_conditions |   aggregation_negative_CI_conditions |
|:----------|------------------------------:|----------------------------------:|----------------------------:|-------------------------:|-------------------------------------:|-------------------------------------:|
| flybrain  |                       0.06389 |                           0.01038 |                       1.352 |                    4.425 |                                    1 |                                    0 |

The strict fanin-to-fanout prototype is a no-go under the pre-declared criteria: no dataset passes all three checks. FlyWire contains a real raw fanin/co-activity signal, but most of it is redundant with fanout and degree. Its candidate occupancy gain peaks below 10%, and only one of six aggregation conditions has a strictly positive 95% CI. The synthetic controls show negligible occupancy gain and mostly neutral or negative aggregation changes.

Large pair-level opportunity lifts do not by themselves establish H1: the selected pairs also have many more shared destinations. Interpret them with the partial correlations and block results, which show that the structural opportunity usually does not convert into reliable additional BlockTask aggregation.
