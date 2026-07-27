# Persistent Block V4 FlyBrain 实验

## 实现

Block V4 保留固定 32-neuron spike block，并增加三个独立编译配置：

```text
BTORCH_BLOCK_EDGE_BUDGET
BTORCH_LONG_SEGMENT_SIZE
BTORCH_TILE_REDUCE_MODE
```

roofline benchmark 对应参数：

```text
--block-edge-budget 0/128/256/512/1024
--long-segment-size 128/256/512/1024/2048
--tile-reduce off/all/hinted
```

实现内容：

- UPDATE 阶段计算普通发放行的 warp active-edge prefix；
- 一个 spike block 按 edge budget 生成多个 logical tasks；
- logical segment 编号压入现有 32-bit task descriptor，无需新增 queue；
- consumer 重新建立 row prefix，只消费当前 logical edge range；
- long-row segment size 编译期参数化；
- 32-edge tile 使用 `match_any` 合并相同 post；
- stats build 直接统计 input edges、global atomics、reduce tasks 和 edges；
- workspace 按最小 128-edge 粒度预留，覆盖全部扫描配置。

`block-edge-budget=0, segment=1024, reduce=off` 保留 V3 路径和行为。

## 环境

```text
server: cccl2
GPU: NVIDIA GeForce RTX 5090, GPU 3
PyTorch: 2.11.0+cu128
driver: 595.71.05
dataset: FlyBrain / FlyWire 783
neurons: 138,639
edges: 15,091,983
batch: 1
timesteps: 128
event rate: 0.02
timing: 10 warmup, 30 repeat, CUDA Event median
```

正式测试期间 GPU 3 空闲。

## 第一轮：普通 BlockTask edge budget

long segment 固定 1024，reduce 关闭。

| reorder | budget | kernel ms | E2E ms |
|---|---:|---:|---:|
| identity | unlimited | 9.544 | 9.544 |
| identity | 256 | 9.695 | 9.695 |
| identity | 512 | 9.242 | 9.242 |
| identity | 1024 | 9.304 | 9.304 |
| global similarity | unlimited | 9.751 | 9.976 |
| global similarity | 256 | 8.685 | 8.904 |
| global similarity | 512 | 8.751 | 8.978 |
| global similarity | 1024 | 9.013 | 9.240 |

budget 切分消除了 similarity 重排造成的重 BlockTask 尾部。总体最佳
budget 为 256；其 E2E 相对 V3 identity 快 6.7%。

动态 task 分布：

```text
original task edges:
  P50 = 567
  P90 = 1,168
  P99 = 2,163
  max = 3,768

budget=256 logical task edges:
  P50/P90/P99/max = 256/256/256/256
  logical tasks / original spike block = 3.03
```

## 第二轮：long segment size

block budget 固定 256，reduce 关闭。

| reorder | segment | kernel ms | E2E ms |
|---|---:|---:|---:|
| identity | 128 | 10.722 | 10.722 |
| identity | 256 | 10.134 | 10.134 |
| identity | 512 | 9.796 | 9.796 |
| identity | 1024 | 9.692 | 9.692 |
| identity | 2048 | 10.249 | 10.249 |
| global similarity | 128 | 9.721 | 9.946 |
| global similarity | 256 | 9.043 | 9.263 |
| global similarity | 512 | 8.550 | 8.775 |
| global similarity | 1024 | 8.691 | 8.910 |
| global similarity | 2048 | 9.275 | 9.499 |

global similarity 的最佳 segment 为 512。相对 1024，它增加 long task
数量：

```text
445,860 -> 579,682
```

但降低单 task 尾部，总时间继续下降。128/256 的 enqueue 和 work-counter
膨胀已经超过收益，因此没有实施可选的 warp cooperative enqueue。

## 第三轮：32-edge tile reduce

固定 budget=256、segment=512。

| reorder | reduce | kernel ms | E2E ms |
|---|---|---:|---:|
| identity | off | 9.798 | 9.798 |
| identity | all | 9.927 | 9.927 |
| identity | hinted | 9.937 | 9.937 |
| global similarity | off | 8.546 | 8.764 |
| global similarity | all | 8.683 | 8.902 |
| global similarity | hinted | 8.680 | 8.899 |
| local cost→similarity | off | 8.993 | 9.139 |
| local cost→similarity | all | 9.113 | 9.264 |
| local cost→similarity | hinted | 9.113 | 9.262 |

stats build 的执行级 atomic 计数：

| reduce | input edges | global atomics | ratio |
|---|---:|---:|---:|
| off | 333,775,945 | 333,775,945 | 1.000000 |
| all | 333,775,945 | 333,358,670 | 0.998750 |
| hinted | 333,775,945 | 333,386,633 | 0.998834 |

task 级 post 重复没有落在相同 32-edge tile 内；all-tile 只减少 0.125%
atomics，无法覆盖 `match_any` 和 peer reduction 成本。按计划停止开发
warp-private hash，最终关闭 reduce。

## 最终配置敏感性

比较：

```text
V3: identity, unlimited block, segment 1024
V4: global similarity, block budget 256, segment 512, reduce off
```

使用 20 warmup、50 repeat：

| event rate | batch | V3 kernel ms | V4 kernel ms | V4 E2E ms |
|---:|---:|---:|---:|---:|
| 0.005 | 1 | 8.550 | 9.578 | 9.797 |
| 0.020 | 1 | 9.543 | 8.538 | 8.757 |
| 0.100 | 1 | 10.547 | 9.643 | 9.869 |
| 0.020 | 4 | 30.413 | 32.199 | 32.892 |

V4 对 batch=1、中高活动率有效：

```text
rate 0.02: kernel +10.5%, E2E +8.2%
rate 0.10: kernel +8.6%, E2E +6.4%
```

低活动率和 batch=4 应继续使用 V3。V4 暂不适合作为所有工作负载的无条件
默认值。

## NCU

Nsight Compute 2026.2，选定指标单次采样：

| 指标 | V3 | V4 |
|---|---:|---:|
| issue active | 6.74% | 6.90% |
| eligible warps / scheduler | 0.08 | 0.09 |
| barrier stall | 35.87% | 18.06% |
| long-scoreboard stall | 40.89% | 49.71% |

budget 切分将 barrier stall 减半，并提高 eligible warp 和 issue active。
long-scoreboard 占比升高，说明负载平衡改善后，global memory/atomic
dependency 成为更显著的剩余瓶颈。NCU replay duration 受计数器采集扰动，
最终延迟以 CUDA Event median 为准。

## 正确性

- budget=512/segment=512：22 个 CUDA persistent 测试通过；
- 最终 budget=256/segment=512：22 个 direct 测试通过；
- tile reduce：9 个 spike-block CUDA 测试通过；
- 762-edge ordinary block 覆盖跨 budget logical-task 切分；
- 2050-edge row 覆盖不同 long segment 数量；
- FlyBrain 单步 PyTorch 对照：

```text
spike mismatch rate = 0
v max absolute error = 0
PSC max absolute error = 7.63e-5
```

## 结论

Block V4 的有效组合是：

```text
reorder = global_similarity
block edge budget = 256
long segment size = 512
tile reduce = off
```

它解决了 similarity 重排原先的重普通任务尾部，并在 FlyBrain batch=1、
event-rate 0.02--0.10 上获得 6.4%--8.2% 端到端加速。

下一步优先级：

1. 根据 batch 和活动率自动选择 V3/V4；
2. 优化 long-scoreboard/global atomic，而不是继续 tile/hash reduce；
3. 若需要 batch=4，再研究跨 batch 的 queue 调度和任务领取粒度。
