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
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1
  ],
  "synthetic_neurons": 8192,
  "synthetic_degree": 64,
  "synthetic_timesteps": 256,
  "flybrain_timesteps": 1024,
  "pair_count": 2000,
  "block_task_sample": 10000,
  "block_size": 32,
  "synthetic_recurrent_strength": 0.6,
  "input_amplitude": 1.1,
  "seed": 20260830
}
```

## Graphs

| dataset   |   neurons |   edges |   density |   mean_fanin |   fanin_cv |   max_fanin |   max_fanout |
|:----------|----------:|--------:|----------:|-------------:|-----------:|------------:|-------------:|
| uniform   |      8192 |  522219 |  0.007782 |        63.75 |     0.1246 |          93 |           64 |
| community |      8192 |  479879 |  0.007151 |        58.58 |     0.1196 |          86 |           64 |
| spatial   |      8192 |  466079 |  0.006945 |        56.89 |     0.1192 |          84 |           64 |

## Measured activity

| dataset   |   input_rate |   timesteps |   actual_firing_rate |   active_neurons |   silent_neuron_fraction |
|:----------|-------------:|------------:|---------------------:|-----------------:|-------------------------:|
| uniform   |        0.001 |         256 |            0.0009222 |             1737 |                  0.788   |
| uniform   |        0.005 |         256 |            0.003921  |             5107 |                  0.3766  |
| uniform   |        0.01  |         256 |            0.007546  |             6787 |                  0.1715  |
| uniform   |        0.02  |         256 |            0.01528   |             7828 |                  0.04443 |
| uniform   |        0.05  |         256 |            0.04787   |             8058 |                  0.01636 |
| uniform   |        0.1   |         256 |            0.1108    |             8018 |                  0.02124 |
| community |        0.001 |         256 |            0.0009131 |             1706 |                  0.7917  |
| community |        0.005 |         256 |            0.003932  |             5110 |                  0.3762  |
| community |        0.01  |         256 |            0.007527  |             6787 |                  0.1715  |
| community |        0.02  |         256 |            0.01501   |             7764 |                  0.05225 |
| community |        0.05  |         256 |            0.04926   |             8070 |                  0.01489 |
| community |        0.1   |         256 |            0.1131    |             8058 |                  0.01636 |
| spatial   |        0.001 |         256 |            0.0008888 |             1656 |                  0.7979  |
| spatial   |        0.005 |         256 |            0.003868  |             5091 |                  0.3785  |
| spatial   |        0.01  |         256 |            0.007479  |             6776 |                  0.1729  |
| spatial   |        0.02  |         256 |            0.01525   |             7810 |                  0.04663 |
| spatial   |        0.05  |         256 |            0.05454   |             8062 |                  0.01587 |
| spatial   |        0.1   |         256 |            0.1257    |             8047 |                  0.0177  |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.001 |                 -0.0201 |                    nan         |            nan        |                    nan        |          nan         |
| uniform   |        0.005 |                 -0.0201 |                      0.01763   |              0.02497  |                      0.01462  |            0.0312    |
| uniform   |        0.01  |                 -0.0201 |                      0.03622   |              0.04883  |                      0.03566  |            0.03969   |
| uniform   |        0.02  |                 -0.0201 |                     -0.01992   |             -0.02106  |                     -0.001456 |            0.001951  |
| uniform   |        0.05  |                 -0.0201 |                      0.03036   |              0.02551  |                      0.05069  |            0.04445   |
| uniform   |        0.1   |                 -0.0201 |                      0.006686  |             -0.009165 |                      0.002363 |           -0.007465  |
| community |        0.001 |                  0.7972 |                    nan         |            nan        |                    nan        |          nan         |
| community |        0.005 |                  0.7972 |                      0.02927   |              0.04634  |                     -0.001193 |           -0.0101    |
| community |        0.01  |                  0.7972 |                      0.05414   |              0.06331  |                      0.06928  |            0.08461   |
| community |        0.02  |                  0.7972 |                      0.00453   |              0.007956 |                      0.006177 |            0.03471   |
| community |        0.05  |                  0.7972 |                      0.008141  |              0.002206 |                      0.004345 |           -0.0001065 |
| community |        0.1   |                  0.7972 |                      0.0354    |              0.03667  |                      0.04947  |            0.003851  |
| spatial   |        0.001 |                  0.9315 |                     -0.00265   |             -0.01501  |                     -0.01325  |            0.00943   |
| spatial   |        0.005 |                  0.9315 |                      0.0207    |              0.03874  |                      0.07252  |            0.1489    |
| spatial   |        0.01  |                  0.9315 |                      0.0001644 |             -0.002349 |                     -0.005124 |           -0.003663  |
| spatial   |        0.02  |                  0.9315 |                      0.01855   |              0.01955  |                      0.03063  |            0.03281   |
| spatial   |        0.05  |                  0.9315 |                      0.0423    |              0.0236   |                      0.02519  |            0.02296   |
| spatial   |        0.1   |                  0.9315 |                      0.03957   |              0.03219  |                      0.02634  |           -0.01645   |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|
| uniform   |        0.001 |        0.006867  |                           0.0001379 |                              0.0001298 |         -8.114e-06 |
| uniform   |        0.005 |       -0.002454  |                           0.0005323 |                              0.0005113 |         -2.099e-05 |
| uniform   |        0.01  |       -7.084e-05 |                           0.001057  |                              0.0009189 |         -0.0001383 |
| uniform   |        0.02  |       -0.0008645 |                           0.002204  |                              0.001796  |         -0.0004078 |
| uniform   |        0.05  |       -0.0005021 |                           0.007002  |                              0.00565   |         -0.001352  |
| uniform   |        0.1   |        0.0001878 |                           0.01586   |                              0.01326   |         -0.0026    |
| community |        0.001 |       -0.003712  |                           0.003211  |                              0.002596  |         -0.0006155 |
| community |        0.005 |       -0.0018    |                           0.009842  |                              0.009089  |         -0.0007534 |
| community |        0.01  |       -0.000712  |                           0.01912   |                              0.01817   |         -0.0009517 |
| community |        0.02  |       -0.004126  |                           0.03816   |                              0.0382    |          3.56e-05  |
| community |        0.05  |       -0.002486  |                           0.1245    |                              0.1227    |         -0.001752  |
| community |        0.1   |        0.000766  |                           0.2521    |                              0.2471    |         -0.00497   |
| spatial   |        0.001 |       -0.002174  |                           0.003187  |                              0.002546  |         -0.0006412 |
| spatial   |        0.005 |       -0.001968  |                           0.01302   |                              0.01213   |         -0.0008907 |
| spatial   |        0.01  |       -0.001075  |                           0.02379   |                              0.02239   |         -0.001404  |
| spatial   |        0.02  |        0.0009947 |                           0.04886   |                              0.04581   |         -0.003049  |
| spatial   |        0.05  |       -0.005709  |                           0.1789    |                              0.1731    |         -0.00587   |
| spatial   |        0.1   |        0.0005736 |                           0.3331    |                              0.3217    |         -0.01133   |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_aggregation_gain_positive   |
|:----------|:-------------------------|:-----------------------------|:-------------------------------|
| community | False                    | False                        | False                          |
| spatial   | False                    | False                        | False                          |
| uniform   | False                    | False                        | False                          |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.
