# Fanout-preserving refinement experiment (v2)

Numerical results only; no images or visualization files were generated.
Profile and evaluation traces use different seeds. Oracle orders are built exclusively from the profile trace.

## Configuration

```json
{
  "datasets": [
    "uniform",
    "community",
    "spatial",
    "flybrain"
  ],
  "target_rates_hz": [
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0
  ],
  "calibration_input_rates": [
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05
  ],
  "calibration_steps": 512,
  "evaluation_steps": 2048,
  "synthetic_neurons": 8192,
  "synthetic_degree": 64,
  "recurrent_strength": 0.1,
  "local_windows": [
    32,
    64,
    128,
    256
  ],
  "joint_window": 128,
  "structural_lambdas": [
    0.0,
    0.25,
    0.5,
    1.0
  ],
  "oracle_lambdas": [
    0.25,
    0.5,
    1.0
  ],
  "block_task_sample": 100000,
  "seed": 20260831
}
```

The common simulator is persistent CUDA v1 LIF+PSC (`dt=1 ms`, soft reset). Synthetic graphs use deterministic balanced 80/20 Dale E/I weights. FlyWire preserves signed relative weights. Every graph is normalized to mean absolute recurrent fanout strength 0.1.

## Graph summary

| dataset   |   neurons |    edges |   mean_fanin |   fanin_cv |
|:----------|----------:|---------:|-------------:|-----------:|
| uniform   |      8192 |   522202 |        63.75 |     0.125  |
| community |      8192 |   479702 |        58.56 |     0.1201 |
| spatial   |      8192 |   466228 |        56.91 |     0.1184 |
| flybrain  |    138639 | 15091983 |       108.9  |     1.482  |

## Calibration and selected controls

| dataset   |   control_input_rate |   population_mean_hz |   active_neuron_mean_hz |   silent_fraction |   selected_targets_hz |
|:----------|---------------------:|---------------------:|------------------------:|------------------:|----------------------:|
| uniform   |               0.0005 |               0.5023 |                   2.216 |         0.7733    |                   0.5 |
| uniform   |               0.001  |               1.013  |                   2.494 |         0.5938    |                   1   |
| uniform   |               0.002  |               1.986  |                   3.114 |         0.3622    |                   2   |
| uniform   |               0.005  |               4.92   |                   5.371 |         0.08398   |                   5   |
| uniform   |               0.01   |               9.859  |                   9.925 |         0.006714  |                  10   |
| uniform   |               0.02   |              19.15   |                  19.16  |         0.0001221 |                  20   |
| uniform   |               0.05   |              47.08   |                  47.08  |         0         |                  50   |
| community |               0.0005 |               0.504  |                   2.212 |         0.7721    |                   0.5 |
| community |               0.001  |               1.025  |                   2.513 |         0.5922    |                   1   |
| community |               0.002  |               1.988  |                   3.11  |         0.361     |                   2   |
| community |               0.005  |               4.978  |                   5.385 |         0.07556   |                   5   |
| community |               0.01   |               9.789  |                   9.858 |         0.006958  |                  10   |
| community |               0.02   |              19.07   |                  19.07  |         0         |                  20   |
| community |               0.05   |              46.73   |                  46.73  |         0         |                  50   |
| spatial   |               0.0005 |               0.4976 |                   2.215 |         0.7754    |                   0.5 |
| spatial   |               0.001  |               0.9983 |                   2.485 |         0.5983    |                   1   |
| spatial   |               0.002  |               2.023  |                   3.131 |         0.3538    |                   2   |
| spatial   |               0.005  |               4.999  |                   5.447 |         0.08228   |                   5   |
| spatial   |               0.01   |               9.794  |                   9.872 |         0.007812  |                  10   |
| spatial   |               0.02   |              19.03   |                  19.03  |         0.0002441 |                  20   |
| spatial   |               0.05   |              46.65   |                  46.65  |         0         |                  50   |
| flybrain  |               0.0005 |               0.4973 |                   2.214 |         0.7754    |                   0.5 |
| flybrain  |               0.001  |               1.005  |                   2.498 |         0.5975    |                   1   |
| flybrain  |               0.002  |               1.987  |                   3.122 |         0.3637    |                   2   |
| flybrain  |               0.005  |               4.743  |                   5.319 |         0.1082    |                   5   |
| flybrain  |               0.01   |              12.7    |                  13.12  |         0.03141   |                  10   |
| flybrain  |               0.02   |              21.82   |                  21.99  |         0.007653  |                  20   |
| flybrain  |               0.05   |              50.1    |                  50.19  |         0.001839  |                  50   |

## Evaluation posterior activity

| dataset   |   control_input_rate |   population_mean_hz |   active_neuron_mean_hz |   silent_fraction |   profile_population_mean_hz |
|:----------|---------------------:|---------------------:|------------------------:|------------------:|-----------------------------:|
| uniform   |               0.001  |               1.009  |                  1.162  |         0.1317    |                       1.004  |
| uniform   |               0.01   |               9.831  |                  9.831  |         0         |                       9.821  |
| uniform   |               0.05   |              46.9    |                 46.9    |         0         |                      46.98   |
| community |               0.001  |               1.007  |                  1.153  |         0.1262    |                       1.001  |
| community |               0.01   |               9.74   |                  9.74   |         0         |                       9.765  |
| community |               0.05   |              46.78   |                 46.78   |         0         |                      46.72   |
| spatial   |               0.001  |               0.9894 |                  1.139  |         0.1317    |                       1.006  |
| spatial   |               0.01   |               9.788  |                  9.788  |         0         |                       9.787  |
| spatial   |               0.05   |              46.71   |                 46.71   |         0         |                      46.63   |
| flybrain  |               0.0005 |               0.4967 |                  0.7782 |         0.3618    |                       0.4974 |
| flybrain  |               0.001  |               0.9998 |                  1.149  |         0.1297    |                       1.003  |
| flybrain  |               0.002  |               1.981  |                  2.022  |         0.02026   |                       1.985  |
| flybrain  |               0.005  |               7.688  |                  7.747  |         0.007559  |                       6.181  |
| flybrain  |               0.01   |              12.86   |                 12.89   |         0.001962  |                      12.94   |
| flybrain  |               0.02   |              22.01   |                 22.04   |         0.001327  |                      21.97   |
| flybrain  |               0.05   |              50.2    |                 50.25   |         0.0009954 |                      50.14   |

## Best method at each measured condition

| dataset   |   population_mean_hz | method             |   occupancy_gain |   static_fanout_change |   aggregation_gain |   atomic_reduction_vs_fanout |   atomic_reduction_ci95_low |   atomic_reduction_ci95_high |
|:----------|---------------------:|:-------------------|-----------------:|-----------------------:|-------------------:|-----------------------------:|----------------------------:|-----------------------------:|
| community |               1.007  | oracle_spike_l0.25 |        0.0002405 |               0.002818 |          3.739e-05 |                    3.748e-05 |                  -0.002335  |                     0.00241  |
| community |               9.74   | oracle_spike_l1    |        0.00132   |               0.003329 |          0.0001368 |                    0.0007985 |                  -0.001792  |                     0.003389 |
| community |              46.78   | oracle_spike_l0.25 |        0.0005379 |               0.003363 |         -0.0002444 |                    0.003503  |                  -0.0002933 |                     0.0073   |
| flybrain  |               0.4967 | joint_fanin_l1     |       -0.0002    |               0.07684  |          9.854e-05 |                    0.007216  |                  -0.008444  |                     0.02287  |
| flybrain  |               0.9998 | joint_fanin_l0.5   |       -0.00054   |               0.0768   |          0.0001716 |                    0.01315   |                  -0.01046   |                     0.03676  |
| flybrain  |               1.981  | joint_fanin_l1     |       -0.0002566 |               0.07684  |          0.0008046 |                    0.005314  |                  -0.02053   |                     0.03116  |
| flybrain  |               7.688  | joint_fanin_l0     |        0.03848   |               0.07638  |          0.03813   |                    0.04683   |                   0.02684   |                     0.06682  |
| flybrain  |              12.86   | joint_fanin_l0.5   |        0.02553   |               0.0768   |          0.03801   |                    0.04886   |                   0.02798   |                     0.06973  |
| flybrain  |              22.01   | joint_fanin_l0     |        0.0142    |               0.07638  |          0.02239   |                    0.04103   |                   0.01929   |                     0.06278  |
| flybrain  |              50.2    | joint_fanin_l1     |        0.004757  |               0.07684  |          0.02643   |                    0.04802   |                   0.0284    |                     0.06765  |
| spatial   |               0.9894 | local_fanin_w32    |        0         |               0        |          0         |                    0         |                  -0.002351  |                     0.002351 |
| spatial   |               9.788  | local_fanin_w32    |        0         |               0        |          0         |                    0         |                  -0.002485  |                     0.002485 |
| spatial   |              46.71   | local_fanin_w32    |        0         |               0        |          0         |                    0         |                  -0.003629  |                     0.003629 |
| uniform   |               1.009  | joint_fanin_l0     |        0.0003606 |               0.02968  |          2.595e-05 |                    2.596e-05 |                  -0.002708  |                     0.00276  |
| uniform   |               9.831  | joint_fanin_l0     |        0.001895  |               0.02968  |          0.0003473 |                    0.0003998 |                  -0.002654  |                     0.003453 |
| uniform   |              46.9    | joint_fanin_l0     |       -0.001166  |               0.02968  |          0.001443  |                    0.002104  |                  -0.002485  |                     0.006694 |

## Best result by method family

| dataset   |   population_mean_hz | method_family   | method             |   occupancy_gain |   atomic_reduction_vs_fanout |   atomic_reduction_ci95_low |
|:----------|---------------------:|:----------------|:-------------------|-----------------:|-----------------------------:|----------------------------:|
| community |               1.007  | joint_fanin     | joint_fanin_l0     |        0.0001202 |                   -1.114e-05 |                  -0.002379  |
| community |               1.007  | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.002352  |
| community |               1.007  | oracle_spike    | oracle_spike_l0.25 |        0.0002405 |                    3.748e-05 |                  -0.002335  |
| community |               9.74   | joint_fanin     | joint_fanin_l1     |        0.00115   |                    0.0007932 |                  -0.001792  |
| community |               9.74   | local_fanin     | local_fanin_w64    |        0.0006951 |                    0.0007873 |                  -0.001799  |
| community |               9.74   | oracle_spike    | oracle_spike_l1    |        0.00132   |                    0.0007985 |                  -0.001792  |
| community |              46.78   | joint_fanin     | joint_fanin_l1     |        0.0004576 |                    0.001117  |                  -0.002688  |
| community |              46.78   | local_fanin     | local_fanin_w64    |        0.0002993 |                    0.001637  |                  -0.002168  |
| community |              46.78   | oracle_spike    | oracle_spike_l0.25 |        0.0005379 |                    0.003503  |                  -0.0002933 |
| flybrain  |               0.4967 | joint_fanin     | joint_fanin_l1     |       -0.0002    |                    0.007216  |                  -0.008444  |
| flybrain  |               0.4967 | local_fanin     | local_fanin_w128   |        0.00015   |                    0.006788  |                  -0.0091    |
| flybrain  |               0.4967 | oracle_spike    | oracle_spike_l1    |       -0.0002071 |                   -0.002662  |                  -0.01895   |
| flybrain  |               0.9998 | joint_fanin     | joint_fanin_l0.5   |       -0.00054   |                    0.01315   |                  -0.01046   |
| flybrain  |               0.9998 | local_fanin     | local_fanin_w64    |       -0.000515  |                    0.01006   |                  -0.0139    |
| flybrain  |               0.9998 | oracle_spike    | oracle_spike_l0.25 |       -0.0005722 |                    0.007803  |                  -0.01624   |
| flybrain  |               1.981  | joint_fanin     | joint_fanin_l1     |       -0.0002566 |                    0.005314  |                  -0.02053   |
| flybrain  |               1.981  | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.02639   |
| flybrain  |               1.981  | oracle_spike    | oracle_spike_l0.5  |       -0.0003775 |                    0.003468  |                  -0.02249   |
| flybrain  |               7.688  | joint_fanin     | joint_fanin_l0     |        0.03848   |                    0.04683   |                   0.02684   |
| flybrain  |               7.688  | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.01949   |
| flybrain  |               7.688  | oracle_spike    | oracle_spike_l0.5  |        0.03913   |                    0.04044   |                   0.02041   |
| flybrain  |              12.86   | joint_fanin     | joint_fanin_l0.5   |        0.02553   |                    0.04886   |                   0.02798   |
| flybrain  |              12.86   | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.02066   |
| flybrain  |              12.86   | oracle_spike    | oracle_spike_l0.5  |        0.0256    |                    0.04141   |                   0.0202    |
| flybrain  |              22.01   | joint_fanin     | joint_fanin_l0     |        0.0142    |                    0.04103   |                   0.01929   |
| flybrain  |              22.01   | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.02179   |
| flybrain  |              22.01   | oracle_spike    | oracle_spike_l0.25 |        0.0145    |                    0.04065   |                   0.01884   |
| flybrain  |              50.2    | joint_fanin     | joint_fanin_l1     |        0.004757  |                    0.04802   |                   0.0284    |
| flybrain  |              50.2    | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.02028   |
| flybrain  |              50.2    | oracle_spike    | oracle_spike_l0.25 |        0.004655  |                    0.04724   |                   0.02749   |
| spatial   |               0.9894 | joint_fanin     | joint_fanin_l0     |       -0.0005506 |                   -0.0002581 |                  -0.002611  |
| spatial   |               0.9894 | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.002351  |
| spatial   |               0.9894 | oracle_spike    | oracle_spike_l0.25 |       -0.001101  |                   -0.0004111 |                  -0.002751  |
| spatial   |               9.788  | joint_fanin     | joint_fanin_l0.25  |       -0.0004521 |                   -0.002013  |                  -0.004522  |
| spatial   |               9.788  | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.002485  |
| spatial   |               9.788  | oracle_spike    | oracle_spike_l0.25 |       -0.0006569 |                   -0.003318  |                  -0.005833  |
| spatial   |              46.71   | joint_fanin     | joint_fanin_l0.5   |       -0.001271  |                   -0.006399  |                  -0.01009   |
| spatial   |              46.71   | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.003629  |
| spatial   |              46.71   | oracle_spike    | oracle_spike_l0.25 |       -0.0006577 |                   -0.007763  |                  -0.01146   |
| uniform   |               1.009  | joint_fanin     | joint_fanin_l0     |        0.0003606 |                    2.596e-05 |                  -0.002708  |
| uniform   |               1.009  | local_fanin     | local_fanin_w32    |        0         |                    0         |                  -0.002733  |
| uniform   |               1.009  | oracle_spike    | oracle_spike_l0.25 |       -0.0005404 |                    2.039e-05 |                  -0.00268   |
| uniform   |               9.831  | joint_fanin     | joint_fanin_l0     |        0.001895  |                    0.0003998 |                  -0.002654  |
| uniform   |               9.831  | local_fanin     | local_fanin_w256   |        0.001479  |                    9.567e-05 |                  -0.002963  |
| uniform   |               9.831  | oracle_spike    | oracle_spike_l0.25 |       -0.0004989 |                   -0.0001282 |                  -0.003171  |
| uniform   |              46.9    | joint_fanin     | joint_fanin_l0     |       -0.001166  |                    0.002104  |                  -0.002485  |
| uniform   |              46.9    | local_fanin     | local_fanin_w128   |       -0.001178  |                    0.0007212 |                  -0.003888  |
| uniform   |              46.9    | oracle_spike    | oracle_spike_l0.25 |       -0.0007848 |                    0.001187  |                  -0.003415  |

## Pure fanout greedy packing (lambda = 0)

| dataset   |   population_mean_hz |   occupancy_gain |   static_fanout_change |   atomic_reduction_vs_fanout |   atomic_reduction_ci95_low |
|:----------|---------------------:|-----------------:|-----------------------:|-----------------------------:|----------------------------:|
| community |               1.007  |        0.0001202 |               0.002725 |                   -1.114e-05 |                   -0.002379 |
| community |               9.74   |        0.001121  |               0.002725 |                    0.0002232 |                   -0.002368 |
| community |              46.78   |        0.0008278 |               0.002725 |                    0.0009399 |                   -0.002863 |
| flybrain  |               0.4967 |       -0.0002071 |               0.07638  |                    0.002687  |                   -0.01314  |
| flybrain  |               0.9998 |       -0.0005722 |               0.07638  |                    0.0004156 |                   -0.02383  |
| flybrain  |               1.981  |       -0.0004948 |               0.07638  |                   -0.002432  |                   -0.02872  |
| flybrain  |               7.688  |        0.03848   |               0.07638  |                    0.04683   |                    0.02684  |
| flybrain  |              12.86   |        0.02523   |               0.07638  |                    0.03856   |                    0.01726  |
| flybrain  |              22.01   |        0.0142    |               0.07638  |                    0.04103   |                    0.01929  |
| flybrain  |              50.2    |        0.004744  |               0.07638  |                    0.04433   |                    0.02444  |
| spatial   |               0.9894 |       -0.0005506 |               0.001068 |                   -0.0002581 |                   -0.002611 |
| spatial   |               9.788  |       -0.0002332 |               0.001068 |                   -0.002076  |                   -0.004587 |
| spatial   |              46.71   |       -0.0008354 |               0.001068 |                   -0.009753  |                   -0.01345  |
| uniform   |               1.009  |        0.0003606 |               0.02968  |                    2.596e-05 |                   -0.002708 |
| uniform   |               9.831  |        0.001895  |               0.02968  |                    0.0003998 |                   -0.002654 |
| uniform   |              46.9    |       -0.001166  |               0.02968  |                    0.002104  |                   -0.002485 |

## Incremental activity contribution over joint lambda = 0

| dataset   |   population_mean_hz | method_family   | method             |   activity_occupancy_gain_vs_joint0 |   activity_atomic_reduction_vs_joint0 |   activity_atomic_reduction_ci95_low |   activity_atomic_reduction_ci95_high |
|:----------|---------------------:|:----------------|:-------------------|------------------------------------:|--------------------------------------:|-------------------------------------:|--------------------------------------:|
| community |               1.007  | joint_fanin     | joint_fanin_l0.5   |                          -0.001141  |                            -0.0001043 |                           -0.002443  |                              0.002235 |
| community |               1.007  | oracle_spike    | oracle_spike_l0.25 |                           0.0001202 |                             4.863e-05 |                           -0.002339  |                              0.002436 |
| community |               9.74   | joint_fanin     | joint_fanin_l1     |                           2.839e-05 |                             0.0005701 |                           -0.002018  |                              0.003159 |
| community |               9.74   | oracle_spike    | oracle_spike_l1    |                           0.0001987 |                             0.0005753 |                           -0.002019  |                              0.00317  |
| community |              46.78   | joint_fanin     | joint_fanin_l1     |                          -0.00037   |                             0.0001776 |                           -0.00363   |                              0.003985 |
| community |              46.78   | oracle_spike    | oracle_spike_l0.25 |                          -0.0002897 |                             0.002566  |                           -0.001233  |                              0.006364 |
| flybrain  |               0.4967 | joint_fanin     | joint_fanin_l1     |                           7.142e-06 |                             0.004541  |                           -0.01089   |                              0.01997  |
| flybrain  |               0.4967 | oracle_spike    | oracle_spike_l1    |                           0         |                            -0.005364  |                           -0.02143   |                              0.0107   |
| flybrain  |               0.9998 | joint_fanin     | joint_fanin_l0.5   |                           3.219e-05 |                             0.01274   |                           -0.01074   |                              0.03622  |
| flybrain  |               0.9998 | oracle_spike    | oracle_spike_l0.25 |                           0         |                             0.007391  |                           -0.01652   |                              0.03131  |
| flybrain  |               1.981  | joint_fanin     | joint_fanin_l1     |                           0.0002383 |                             0.007727  |                           -0.01788   |                              0.03333  |
| flybrain  |               1.981  | oracle_spike    | oracle_spike_l0.5  |                           0.0001173 |                             0.005886  |                           -0.01983   |                              0.03161  |
| flybrain  |               7.688  | joint_fanin     | joint_fanin_l0.25  |                           0.0007087 |                            -0.001623  |                           -0.02403   |                              0.02078  |
| flybrain  |               7.688  | oracle_spike    | oracle_spike_l0.5  |                           0.0006352 |                            -0.006707  |                           -0.02911   |                              0.0157   |
| flybrain  |              12.86   | joint_fanin     | joint_fanin_l0.5   |                           0.0002952 |                             0.01071   |                           -0.01239   |                              0.03382  |
| flybrain  |              12.86   | oracle_spike    | oracle_spike_l0.5  |                           0.0003683 |                             0.002972  |                           -0.02049   |                              0.02643  |
| flybrain  |              22.01   | joint_fanin     | joint_fanin_l0.25  |                           5.316e-05 |                            -0.004395  |                           -0.02808   |                              0.01929  |
| flybrain  |              22.01   | oracle_spike    | oracle_spike_l0.25 |                           0.0002955 |                            -0.0003943 |                           -0.02399   |                              0.0232   |
| flybrain  |              50.2    | joint_fanin     | joint_fanin_l1     |                           1.256e-05 |                             0.003863  |                           -0.0172    |                              0.02492  |
| flybrain  |              50.2    | oracle_spike    | oracle_spike_l0.25 |                          -8.92e-05  |                             0.003045  |                           -0.01815   |                              0.02424  |
| spatial   |               0.9894 | joint_fanin     | joint_fanin_l1     |                          -6.118e-05 |                            -1.699e-05 |                           -0.002357  |                              0.002323 |
| spatial   |               0.9894 | oracle_spike    | oracle_spike_l0.25 |                          -0.0005503 |                            -0.0001529 |                           -0.002493  |                              0.002188 |
| spatial   |               9.788  | joint_fanin     | joint_fanin_l0.25  |                          -0.000219  |                             6.272e-05 |                           -0.002461  |                              0.002586 |
| spatial   |               9.788  | oracle_spike    | oracle_spike_l0.25 |                          -0.0004238 |                            -0.001239  |                           -0.00377   |                              0.001292 |
| spatial   |              46.71   | joint_fanin     | joint_fanin_l0.5   |                          -0.0004358 |                             0.003322  |                           -0.0003606 |                              0.007004 |
| spatial   |              46.71   | oracle_spike    | oracle_spike_l0.25 |                           0.0001778 |                             0.00197   |                           -0.001718  |                              0.005659 |
| uniform   |               1.009  | joint_fanin     | joint_fanin_l0.25  |                          -0.001081  |                            -2.04e-05  |                           -0.002723  |                              0.002682 |
| uniform   |               1.009  | oracle_spike    | oracle_spike_l0.25 |                          -0.0009007 |                            -5.562e-06 |                           -0.002707  |                              0.002696 |
| uniform   |               9.831  | joint_fanin     | joint_fanin_l0.25  |                           0.0007824 |                            -0.0003951 |                           -0.003462  |                              0.002672 |
| uniform   |               9.831  | oracle_spike    | oracle_spike_l0.25 |                          -0.002389  |                            -0.0005282 |                           -0.003578  |                              0.002521 |
| uniform   |              46.9    | joint_fanin     | joint_fanin_l0.25  |                          -0.0003278 |                            -0.0009259 |                           -0.005529  |                              0.003677 |
| uniform   |              46.9    | oracle_spike    | oracle_spike_l0.25 |                           0.0003815 |                            -0.0009193 |                           -0.00552   |                              0.003681 |

## Oracle profile stability

| dataset   |   population_mean_hz |   profile_nonzero_fraction |   evaluation_nonzero_fraction |   persistence_given_profile |   positive_support_jaccard |   spearman_all |   spearman_any_positive |
|:----------|---------------------:|---------------------------:|------------------------------:|----------------------------:|---------------------------:|---------------:|------------------------:|
| uniform   |               1.009  |                  0.002258  |                     0.002127  |                    0        |                   0        |     -0.002196  |                -0.8594  |
| uniform   |               9.831  |                  0.1792    |                     0.1776    |                    0.1803   |                   0.09957  |      0.004205  |                -0.665   |
| uniform   |              46.9    |                  0.9872    |                     0.9864    |                    0.9865   |                   0.9742   |      0.01796   |                 0.01712 |
| community |               1.007  |                  0.001965  |                     0.002107  |                    0        |                   0        |     -0.002039  |                -0.8594  |
| community |               9.74   |                  0.1777    |                     0.1752    |                    0.1766   |                   0.09765  |      0.0011    |                -0.6757  |
| community |              46.78   |                  0.9863    |                     0.9863    |                    0.9865   |                   0.9734   |      0.02379   |                 0.02269 |
| spatial   |               0.9894 |                  0.002197  |                     0.002026  |                    0.004587 |                   0.002392 |      0.002683  |                -0.8498  |
| spatial   |               9.788  |                  0.1791    |                     0.1771    |                    0.179    |                   0.09887  |      0.00175   |                -0.6722  |
| spatial   |              46.71   |                  0.9853    |                     0.9859    |                    0.9859   |                   0.9718   |      0.02358   |                 0.02272 |
| flybrain  |               0.4967 |                  0.0004943 |                     0.0005245 |                    0        |                   0        |     -0.0005095 |                -0.8635  |
| flybrain  |               0.9998 |                  0.001947  |                     0.002139  |                    0        |                   0        |     -0.002045  |                -0.8594  |
| flybrain  |               1.981  |                  0.007303  |                     0.007908  |                    0.009669 |                   0.004664 |      0.001685  |                -0.8494  |
| flybrain  |               7.688  |                  0.05733   |                     0.06427   |                    0.2267   |                   0.1197   |      0.1503    |                -0.7842  |
| flybrain  |              12.86   |                  0.1834    |                     0.1855    |                    0.2942   |                   0.1713   |      0.09749   |                -0.6772  |
| flybrain  |              22.01   |                  0.4912    |                     0.4927    |                    0.549    |                   0.3776   |      0.07968   |                -0.3226  |
| flybrain  |              50.2    |                  0.9581    |                     0.9582    |                    0.9696   |                   0.9409   |      0.199     |                 0.1695  |

## Decision summary

| dataset   |   joint0_median_atomic_reduction_vs_fanout |   best_structural_median_increment_vs_joint0 |   best_structural_median_lower_CI |   best_oracle_median_reduction_vs_fanout |   best_oracle_median_increment_vs_joint0 |   best_oracle_median_lower_CI | classification                            |
|:----------|-------------------------------------------:|---------------------------------------------:|----------------------------------:|-----------------------------------------:|-----------------------------------------:|------------------------------:|:------------------------------------------|
| uniform   |                                  0.0003998 |                                   -0.0003951 |                         -0.003462 |                                2.039e-05 |                               -0.0005282 |                     -0.003578 | stop activity-aware; retain fanout greedy |
| community |                                  0.0002232 |                                    0.0001776 |                         -0.002443 |                                0.0007985 |                                0.0005753 |                     -0.002019 | stop activity-aware; retain fanout greedy |
| spatial   |                                 -0.002076  |                                    6.272e-05 |                         -0.002357 |                               -0.003318  |                               -0.0001529 |                     -0.002493 | stop activity-aware; retain fanout greedy |
| flybrain  |                                  0.03856   |                                    0.004541  |                         -0.0172   |                                0.04044   |                                0.002972  |                     -0.02049  | stop activity-aware; retain fanout greedy |

Atomic counts are ratio-of-means estimates from uniformly sampled active BlockTasks; reported intervals propagate independent candidate/baseline standard errors. Occupancy and active-block counts are exact over the entire evaluation trace.

The decision separates fanout greedy packing (`lambda=0`) from the incremental contribution of structural or measured activity affinity. This prevents a stronger fanout objective from being misattributed to fanin or spike profiling.

## Data interpretation

None of the 64 local-fanin window candidates produced a reliable atomic reduction over the existing fanout order (0 had a positive 95% CI lower bound).

After controlling for the stronger pure-fanout greedy packer, 0 of 32 condition-level best structural/oracle candidates had a positive 95% CI lower bound. The activity-aware headroom is therefore unsupported.

A separate fanout-only result remains actionable: on FlyWire from 7.69 to 50.2 Hz, lambda=0 reduced the estimated atomic count by 3.86% to 4.68% with positive 95% CI lower bounds, while static fanout aggregation increased by 7.64 percentage points. This is an improved fanout packing objective, not evidence for fanin or activity-aware refinement.

Recommendation: stop the fanin/activity-aware branch and retain lambda=0 fanout-union greedy packing as a separate FlyWire preprocessing candidate. Validate that candidate in the real kernel before adoption; this experiment estimates structural execution cost and does not measure kernel runtime.
