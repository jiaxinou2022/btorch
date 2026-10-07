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
  "synthetic_recurrent_strength": 0.3,
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
| flybrain  |        0.001 |         128 |              0.01294 |            23718 |                  0.8289  |
| flybrain  |        0.005 |         128 |              0.01696 |            58112 |                  0.5808  |
| flybrain  |        0.01  |         128 |              0.02207 |            80767 |                  0.4174  |
| flybrain  |        0.02  |         128 |              0.03056 |           104409 |                  0.2469  |
| flybrain  |        0.05  |         128 |              0.05532 |           128006 |                  0.0767  |
| flybrain  |        0.1   |         128 |              0.1     |           134955 |                  0.02657 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| flybrain  |        0.001 |                  0.8596 |                        0.1493  |               0.3982  |                      0.02025  |             0.02068  |
| flybrain  |        0.005 |                  0.8596 |                        0.1334  |               0.2575  |                      0.01417  |            -0.09978  |
| flybrain  |        0.01  |                  0.8596 |                        0.0845  |               0.1102  |                      0.009994 |            -0.006415 |
| flybrain  |        0.02  |                  0.8596 |                        0.1158  |               0.1585  |                      0.01727  |             0.01324  |
| flybrain  |        0.05  |                  0.8596 |                        0.03771 |               0.03734 |                      0.05569  |             0.07937  |
| flybrain  |        0.1   |                  0.8596 |                        0.04502 |               0.03072 |                      0.01548  |             0.01197  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |   aggregation_gain_ci95_low |   aggregation_gain_ci95_high |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|----------------------------:|-----------------------------:|
| flybrain  |        0.001 |         0.07758  |                             0.1294  |                                0.1509  |           0.0215   |                   0.006614  |                      0.03639 |
| flybrain  |        0.005 |         0.05632  |                             0.1105  |                                0.135   |           0.02451  |                   0.01031   |                      0.03871 |
| flybrain  |        0.01  |         0.04005  |                             0.1068  |                                0.1242  |           0.01735  |                   0.001802  |                      0.03289 |
| flybrain  |        0.02  |         0.02878  |                             0.1047  |                                0.1119  |           0.007248 |                  -0.007646  |                      0.02214 |
| flybrain  |        0.05  |         0.01049  |                             0.09147 |                                0.08926 |          -0.002211 |                  -0.01605   |                      0.01162 |
| flybrain  |        0.1   |         0.002439 |                             0.0761  |                                0.08684 |           0.01074  |                   0.0008026 |                      0.02068 |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_gain_lower_CI_positive   |
|:----------|:-------------------------|:-----------------------------|:----------------------------|
| flybrain  | False                    | False                        | True                        |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.

## Data-first interpretation

| dataset   |   median_raw_fanin_spike_corr |   median_partial_fanin_spike_corr |   median_occupancy_gain_pct |   max_occupancy_gain_pct |   aggregation_positive_CI_conditions |   aggregation_negative_CI_conditions |
|:----------|------------------------------:|----------------------------------:|----------------------------:|-------------------------:|-------------------------------------:|-------------------------------------:|
| flybrain  |                        0.1001 |                           0.01637 |                       3.441 |                    7.758 |                                    4 |                                    0 |

The strict fanin-to-fanout prototype is a no-go under the pre-declared criteria: no dataset passes all three checks. FlyWire contains a real raw fanin/co-activity signal, but most of it is redundant with fanout and degree. Its candidate occupancy gain peaks below 10%, and only one of six aggregation conditions has a strictly positive 95% CI. The synthetic controls show negligible occupancy gain and mostly neutral or negative aggregation changes.

Large pair-level opportunity lifts do not by themselves establish H1: the selected pairs also have many more shared destinations. Interpret them with the partial correlations and block results, which show that the structural opportunity usually does not convert into reliable additional BlockTask aggregation.
