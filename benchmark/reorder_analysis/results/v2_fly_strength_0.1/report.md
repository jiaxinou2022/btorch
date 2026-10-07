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
  "synthetic_recurrent_strength": 0.1,
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
| flybrain  |        0.001 |         128 |             0.001008 |            16791 |                 0.8789   |
| flybrain  |        0.005 |         128 |             0.004765 |            62323 |                 0.5505   |
| flybrain  |        0.01  |         128 |             0.01151  |            95172 |                 0.3135   |
| flybrain  |        0.02  |         128 |             0.02141  |           121509 |                 0.1236   |
| flybrain  |        0.05  |         128 |             0.04998  |           136812 |                 0.01318  |
| flybrain  |        0.1   |         128 |             0.09905  |           138353 |                 0.002063 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| flybrain  |        0.001 |                  0.8596 |                       0.002243 |             -0.1411   |                      0.002925 |            nan       |
| flybrain  |        0.005 |                  0.8596 |                       0.02184  |              0.05308  |                     -0.03773  |             -0.1014  |
| flybrain  |        0.01  |                  0.8596 |                       0.07559  |              0.1009   |                      0.005929 |              0.0215  |
| flybrain  |        0.02  |                  0.8596 |                       0.07145  |              0.07429  |                      0.03325  |              0.0605  |
| flybrain  |        0.05  |                  0.8596 |                       0.002778 |             -0.002411 |                      0.002253 |              0.01154 |
| flybrain  |        0.1   |                  0.8596 |                       0.07661  |              0.04982  |                      0.01822  |              0.02295 |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |   aggregation_gain_ci95_low |   aggregation_gain_ci95_high |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|----------------------------:|-----------------------------:|
| flybrain  |        0.001 |        0.0001135 |                           0.0003961 |                              0.0002364 |         -0.0001596 |                  -0.0004251 |                    0.0001058 |
| flybrain  |        0.005 |        0.002329  |                           0.001741  |                              0.005678  |          0.003937  |                  -0.001062  |                    0.008936  |
| flybrain  |        0.01  |        0.01452   |                           0.08171   |                              0.1017    |          0.02004   |                  -0.001501  |                    0.04159   |
| flybrain  |        0.02  |        0.009881  |                           0.0687    |                              0.08274   |          0.01404   |                  -0.005825  |                    0.0339    |
| flybrain  |        0.05  |        0.003591  |                           0.05117   |                              0.06305   |          0.01188   |                  -0.001836  |                    0.0256    |
| flybrain  |        0.1   |        0.0003711 |                           0.05702   |                              0.06119   |          0.004165  |                  -0.005238  |                    0.01357   |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_gain_lower_CI_positive   |
|:----------|:-------------------------|:-----------------------------|:----------------------------|
| flybrain  | False                    | False                        | False                       |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.

## Data-first interpretation

| dataset   |   median_raw_fanin_spike_corr |   median_partial_fanin_spike_corr |   median_occupancy_gain_pct |   max_occupancy_gain_pct |   aggregation_positive_CI_conditions |   aggregation_negative_CI_conditions |
|:----------|------------------------------:|----------------------------------:|----------------------------:|-------------------------:|-------------------------------------:|-------------------------------------:|
| flybrain  |                       0.04664 |                          0.004427 |                       0.296 |                    1.452 |                                    0 |                                    0 |

The strict fanin-to-fanout prototype is a no-go under the pre-declared criteria: no dataset passes all three checks. FlyWire contains a real raw fanin/co-activity signal, but most of it is redundant with fanout and degree. Its candidate occupancy gain peaks below 10%, and only one of six aggregation conditions has a strictly positive 95% CI. The synthetic controls show negligible occupancy gain and mostly neutral or negative aggregation changes.

Large pair-level opportunity lifts do not by themselves establish H1: the selected pairs also have many more shared destinations. Interpret them with the partial correlations and block results, which show that the structural opportunity usually does not convert into reliable additional BlockTask aggregation.
