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
    "spatial",
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
  "flybrain_timesteps": 1024,
  "pair_count": 100000,
  "block_task_sample": 10000,
  "block_size": 32,
  "synthetic_recurrent_strength": 0.6,
  "input_amplitude": 1.1,
  "seed": 20260830
}
```

The simulator is the repository's persistent CUDA v1 LIF+PSC path (`dt=1`, threshold 1, soft reset). External event amplitude is 1.1, so one event leaves only a 0.1 post-reset residual. Synthetic graphs use a deterministic 80/20 Dale E/I split with inhibitory edges 4x stronger and mean recurrent strength 0.6. FlyWire retains signed relative synapse counts but is normalized to the same mean absolute fanout strength; this is a common topology workload, not a faithful Shiu FlyBrain reproduction.

## Graphs

| dataset   |   neurons |    edges |   density |   mean_fanin |   fanin_cv |   max_fanin |   max_fanout |
|:----------|----------:|---------:|----------:|-------------:|-----------:|------------:|-------------:|
| uniform   |      8192 |   522219 | 0.007782  |        63.75 |     0.1246 |          93 |           64 |
| community |      8192 |   479879 | 0.007151  |        58.58 |     0.1196 |          86 |           64 |
| spatial   |      8192 |   466079 | 0.006945  |        56.89 |     0.1192 |          84 |           64 |
| flybrain  |    138639 | 15091983 | 0.0007852 |       108.9  |     1.482  |       10356 |         9783 |

## Measured activity

| dataset   |   input_rate |   timesteps |   actual_firing_rate |   active_neurons |   silent_neuron_fraction |
|:----------|-------------:|------------:|---------------------:|-----------------:|-------------------------:|
| uniform   |        0.001 |        2048 |            0.0009105 |             6883 |                0.1598    |
| uniform   |        0.005 |        2048 |            0.003934  |             8183 |                0.001099  |
| uniform   |        0.01  |        2048 |            0.007537  |             8192 |                0         |
| uniform   |        0.02  |        2048 |            0.01529   |             8192 |                0         |
| uniform   |        0.05  |        2048 |            0.04823   |             8190 |                0.0002441 |
| uniform   |        0.1   |        2048 |            0.1102    |             8142 |                0.006104  |
| community |        0.001 |        2048 |            0.0009    |             6856 |                0.1631    |
| community |        0.005 |        2048 |            0.003865  |             8187 |                0.0006104 |
| community |        0.01  |        2048 |            0.007487  |             8192 |                0         |
| community |        0.02  |        2048 |            0.01538   |             8192 |                0         |
| community |        0.05  |        2048 |            0.05077   |             8192 |                0         |
| community |        0.1   |        2048 |            0.1152    |             8158 |                0.00415   |
| spatial   |        0.001 |        2048 |            0.000887  |             6900 |                0.1577    |
| spatial   |        0.005 |        2048 |            0.003883  |             8184 |                0.0009766 |
| spatial   |        0.01  |        2048 |            0.007421  |             8191 |                0.0001221 |
| spatial   |        0.02  |        2048 |            0.0155    |             8192 |                0         |
| spatial   |        0.05  |        2048 |            0.05689   |             8182 |                0.001221  |
| spatial   |        0.1   |        2048 |            0.1324    |             8144 |                0.005859  |
| flybrain  |        0.001 |        1024 |            0.05118   |            58263 |                0.5798    |
| flybrain  |        0.005 |        1024 |            0.05467   |            92879 |                0.3301    |
| flybrain  |        0.01  |        1024 |            0.05744   |           105610 |                0.2382    |
| flybrain  |        0.02  |        1024 |            0.06494   |           114926 |                0.171     |
| flybrain  |        0.05  |        1024 |            0.0843    |           126127 |                0.09025   |
| flybrain  |        0.1   |        1024 |            0.1193    |           132144 |                0.04685   |

## H1: topology predicts co-activity

| dataset   |   input_rate |   spearman_fanin_fanout |   spearman_fanin_spike_jaccard |   spearman_fanin_lift |   partial_fanin_spike_jaccard |   partial_fanin_lift |
|:----------|-------------:|------------------------:|-------------------------------:|----------------------:|------------------------------:|---------------------:|
| uniform   |        0.001 |               0.0009977 |                     -0.001189  |            -0.001406  |                     0.0006704 |            0.001664  |
| uniform   |        0.005 |               0.0009977 |                     -0.003402  |            -0.003353  |                    -0.005322  |           -0.003384  |
| uniform   |        0.01  |               0.0009977 |                      0.003481  |             0.003446  |                     0.001366  |            0.001062  |
| uniform   |        0.02  |               0.0009977 |                      0.004079  |             0.003549  |                     0.004975  |            0.005197  |
| uniform   |        0.05  |               0.0009977 |                      0.00668   |             0.0003723 |                     0.006788  |           -0.00211   |
| uniform   |        0.1   |               0.0009977 |                      0.009949  |             0.009465  |                     0.01122   |            7.744e-05 |
| community |        0.001 |               0.7895    |                     -0.0007209 |            -0.0008969 |                    -0.005526  |           -0.007228  |
| community |        0.005 |               0.7895    |                      0.001039  |             0.001088  |                     0.0008064 |            0.0009254 |
| community |        0.01  |               0.7895    |                      0.01184   |             0.01182   |                     0.005969  |            0.006792  |
| community |        0.02  |               0.7895    |                      0.01223   |             0.01011   |                     0.003626  |            0.002005  |
| community |        0.05  |               0.7895    |                      0.03826   |             0.0391    |                     0.0409    |            0.0151    |
| community |        0.1   |               0.7895    |                      0.03207   |             0.0613    |                     0.05507   |            0.01171   |
| spatial   |        0.001 |               0.9308    |                     -0.001584  |            -0.00192   |                    -0.001385  |           -0.001438  |
| spatial   |        0.005 |               0.9308    |                      0.002866  |             0.002859  |                    -0.004622  |           -0.004612  |
| spatial   |        0.01  |               0.9308    |                     -0.0001019 |            -0.0002504 |                    -0.002884  |           -0.004412  |
| spatial   |        0.02  |               0.9308    |                      0.00987   |             0.006744  |                     0.005537  |            0.004799  |
| spatial   |        0.05  |               0.9308    |                      0.04644   |             0.05091   |                     0.05245   |            0.01101   |
| spatial   |        0.1   |               0.9308    |                      0.03515   |             0.07784   |                     0.0587    |            0.004268  |
| flybrain  |        0.001 |               0.8666    |                      0.1208    |             0.2058    |                     0.03752   |            0.04176   |
| flybrain  |        0.005 |               0.8666    |                      0.06952   |             0.1075    |                     0.03362   |            0.007181  |
| flybrain  |        0.01  |               0.8666    |                      0.04778   |             0.0707    |                     0.03287   |            0.01076   |
| flybrain  |        0.02  |               0.8666    |                      0.03407   |             0.02954   |                     0.03136   |            0.006381  |
| flybrain  |        0.05  |               0.8666    |                     -0.001589  |            -0.003462  |                     0.02958   |            0.01231   |
| flybrain  |        0.1   |               0.8666    |                     -0.02819   |            -0.01447   |                     0.02184   |            0.00742   |

## H2/H3: candidate relative to fanout-only

| dataset   |   input_rate |   occupancy_gain |   weighted_aggregation_ratio_fanout |   weighted_aggregation_ratio_candidate |   aggregation_gain |   aggregation_gain_ci95_low |   aggregation_gain_ci95_high |
|:----------|-------------:|-----------------:|------------------------------------:|---------------------------------------:|-------------------:|----------------------------:|-----------------------------:|
| uniform   |        0.001 |        0.002793  |                           0.0001255 |                              0.0001159 |         -9.604e-06 |                  -5.721e-05 |                    3.8e-05   |
| uniform   |        0.005 |        0.0001769 |                           0.0006069 |                              0.000486  |         -0.0001209 |                  -0.0002225 |                   -1.924e-05 |
| uniform   |        0.01  |       -5.321e-05 |                           0.001055  |                              0.0009555 |         -9.961e-05 |                  -0.0002297 |                    3.044e-05 |
| uniform   |        0.02  |       -0.0007927 |                           0.002222  |                              0.001806  |         -0.0004161 |                  -0.0005956 |                   -0.0002367 |
| uniform   |        0.05  |       -0.000293  |                           0.006922  |                              0.005678  |         -0.001244  |                  -0.001509  |                   -0.0009795 |
| uniform   |        0.1   |        0.0002953 |                           0.01596   |                              0.01321   |         -0.00275   |                  -0.003078  |                   -0.002422  |
| community |        0.001 |        0         |                           0.002346  |                              0.00237   |          2.447e-05 |                  -0.0005457 |                    0.0005946 |
| community |        0.005 |        0.0006391 |                           0.009755  |                              0.009649  |         -0.0001068 |                  -0.001198  |                    0.0009841 |
| community |        0.01  |        0.0001611 |                           0.01947   |                              0.01839   |         -0.001087  |                  -0.002585  |                    0.0004115 |
| community |        0.02  |       -0.001802  |                           0.03997   |                              0.03653   |         -0.003439  |                  -0.005513  |                   -0.001365  |
| community |        0.05  |        0.0002915 |                           0.1282    |                              0.1261    |         -0.002133  |                  -0.005669  |                    0.001402  |
| community |        0.1   |        0.000349  |                           0.2548    |                              0.2509    |         -0.00388   |                  -0.007927  |                    0.0001676 |
| spatial   |        0.001 |        0.001706  |                           0.003086  |                              0.00331   |          0.0002243 |                  -0.0004904 |                    0.000939  |
| spatial   |        0.005 |        0.00031   |                           0.01185   |                              0.01202   |          0.0001688 |                  -0.001238  |                    0.001575  |
| spatial   |        0.01  |       -0.0001263 |                           0.0234    |                              0.02377   |          0.0003685 |                  -0.001432  |                    0.002169  |
| spatial   |        0.02  |       -0.001229  |                           0.04952   |                              0.04757   |         -0.001952  |                  -0.004536  |                    0.0006321 |
| spatial   |        0.05  |       -0.0002801 |                           0.1889    |                              0.1794    |         -0.009474  |                  -0.01498   |                   -0.003969  |
| spatial   |        0.1   |        0.0001608 |                           0.3431    |                              0.3365    |         -0.006576  |                  -0.01302   |                   -0.000127  |
| flybrain  |        0.001 |        0.05336   |                           0.1552    |                              0.1746    |          0.01946   |                   0.007005  |                    0.03192   |
| flybrain  |        0.005 |        0.04703   |                           0.156     |                              0.1673    |          0.01123   |                  -7.661e-05 |                    0.02253   |
| flybrain  |        0.01  |        0.04225   |                           0.1519    |                              0.161     |          0.009152  |                  -0.002503  |                    0.02081   |
| flybrain  |        0.02  |        0.0299    |                           0.1509    |                              0.1506    |         -0.0003589 |                  -0.01195   |                    0.01123   |
| flybrain  |        0.05  |        0.01469   |                           0.1434    |                              0.1465    |          0.003089  |                  -0.009067  |                    0.01524   |
| flybrain  |        0.1   |        0.005278  |                           0.1186    |                              0.1266    |          0.007972  |                  -0.001815  |                    0.01776   |

## Pre-declared decision checks

| dataset   | H1_partial_corr_gt_.05   | H2_occupancy_gain_gt_10pct   | H3_gain_lower_CI_positive   |
|:----------|:-------------------------|:-----------------------------|:----------------------------|
| community | False                    | False                        | False                       |
| flybrain  | False                    | False                        | False                       |
| spatial   | False                    | False                        | False                       |
| uniform   | False                    | False                        | False                       |

These checks are diagnostics, not significance tests. H1 uses a modest partial-correlation threshold because finite low-rate traces make pairwise lift highly zero-inflated. H2 and H3 follow the plan's 10% occupancy and positive aggregation-improvement criteria.

## Data-first interpretation

| dataset   |   median_raw_fanin_spike_corr |   median_partial_fanin_spike_corr |   median_occupancy_gain_pct |   max_occupancy_gain_pct |   aggregation_positive_CI_conditions |   aggregation_negative_CI_conditions |
|:----------|------------------------------:|----------------------------------:|----------------------------:|-------------------------:|-------------------------------------:|-------------------------------------:|
| community |                      0.01204  |                          0.004797 |                    0.02263  |                  0.06391 |                                    0 |                                    1 |
| flybrain  |                      0.04093  |                          0.03212  |                    3.607    |                  5.336   |                                    1 |                                    0 |
| spatial   |                      0.006368 |                          0.002076 |                    0.001727 |                  0.1706  |                                    0 |                                    2 |
| uniform   |                      0.00378  |                          0.00317  |                    0.006185 |                  0.2793  |                                    0 |                                    4 |

The strict fanin-to-fanout prototype is a no-go under the pre-declared criteria: no dataset passes all three checks. FlyWire contains a real raw fanin/co-activity signal, but most of it is redundant with fanout and degree. Its candidate occupancy gain peaks below 10%, and only one of six aggregation conditions has a strictly positive 95% CI. The synthetic controls show negligible occupancy gain and mostly neutral or negative aggregation changes.

Large pair-level opportunity lifts do not by themselves establish H1: the selected pairs also have many more shared destinations. Interpret them with the partial correlations and block results, which show that the structural opportunity usually does not convert into reliable additional BlockTask aggregation.
