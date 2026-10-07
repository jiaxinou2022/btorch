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
  "synthetic_recurrent_strength": 0.05,
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
| flybrain  |        0.001 |         128 |             0.001007 |            16792 |                0.8789    |
| flybrain  |        0.005 |         128 |             0.004998 |            65331 |                0.5288    |
| flybrain  |        0.01  |         128 |             0.009883 |            99229 |                0.2843    |
| flybrain  |        0.02  |         128 |             0.02023  |           126107 |                0.09039   |
| flybrain  |        0.05  |         128 |             0.05025  |           138043 |                0.004299  |
| flybrain  |        0.1   |         128 |             0.1006   |           138607 |                0.0002308 |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| flybrain  |        0.001 |                  0.8596 |                       0.002243 |             -0.154    |                      0.004205 |           nan        |
| flybrain  |        0.005 |                  0.8596 |                       0.03056  |              0.06591  |                     -0.03822  |            -0.07856  |
| flybrain  |        0.01  |                  0.8596 |                       0.01815  |              0.0213   |                      0.01532  |             0.02209  |
| flybrain  |        0.02  |                  0.8596 |                       0.08539  |              0.09133  |                      0.03829  |             0.05871  |
| flybrain  |        0.05  |                  0.8596 |                       0.009005 |              0.001952 |                     -0.02083  |            -0.005015 |
| flybrain  |        0.1   |                  0.8596 |                       0.06822  |              0.04993  |                      0.03648  |             0.04995  |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |   aggregation_gain_ci95_low |   aggregation_gain_ci95_high |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|----------------------------:|-----------------------------:|
| flybrain  |        0.001 |        0.0001136 |                           0.0004823 |                              0.0003066 |         -0.0001758 |                  -0.0004843 |                    0.0001327 |
| flybrain  |        0.005 |        0.002998  |                           0.001882  |                              0.001254  |         -0.0006273 |                  -0.001215  |                   -4.006e-05 |
| flybrain  |        0.01  |        0.0003311 |                           0.003695  |                              0.004451  |          0.0007557 |                  -0.002387  |                    0.003898  |
| flybrain  |        0.02  |        0.003148  |                           0.04244   |                              0.03689   |         -0.005548  |                  -0.01949   |                    0.008395  |
| flybrain  |        0.05  |        0.002105  |                           0.04171   |                              0.04882   |          0.007112  |                  -0.005423  |                    0.01965   |
| flybrain  |        0.1   |       -0.0002184 |                           0.05236   |                              0.04952   |         -0.002845  |                  -0.01116   |                    0.005468  |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_gain_lower_CI_positive   |
|:----------|:-------------------------|:-----------------------------|:----------------------------|
| flybrain  | False                    | False                        | False                       |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.

## Data-first interpretation

| dataset   |   median_raw_fanin_spike_corr |   median_partial_fanin_spike_corr |   median_occupancy_gain_pct |   max_occupancy_gain_pct |   aggregation_positive_CI_conditions |   aggregation_negative_CI_conditions |
|:----------|------------------------------:|----------------------------------:|----------------------------:|-------------------------:|-------------------------------------:|-------------------------------------:|
| flybrain  |                       0.02435 |                          0.009762 |                      0.1218 |                   0.3148 |                                    0 |                                    1 |

The strict fanin-to-fanout prototype is a no-go under the pre-declared criteria: no dataset passes all three checks. FlyWire contains a real raw fanin/co-activity signal, but most of it is redundant with fanout and degree. Its candidate occupancy gain peaks below 10%, and only one of six aggregation conditions has a strictly positive 95% CI. The synthetic controls show negligible occupancy gain and mostly neutral or negative aggregation changes.

Large pair-level opportunity lifts do not by themselves establish H1: the selected pairs also have many more shared destinations. Interpret them with the partial correlations and block results, which show that the structural opportunity usually does not convert into reliable additional BlockTask aggregation.
