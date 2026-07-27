# Persistent SNN 物理重排 MVP 实验

## 实现范围

本轮按 `block_preprocess_plan.md` 完成了不修改 persistent kernel 的 MVP：

- `new_to_old` / `old_to_new` 双向映射；
- GPU tensor 上的 CSR row 物理搬运和 post index 重映射；
- event index、state、dense/event output 的正向与逆向重排；
- identity、全局/局部 fanout 排序、粗分桶、成本均衡、dominant-post
  similarity 以及 cost/similarity 组合策略；
- exact dominant post block 统计，空行使用 `-1`，平局取最小 block；
- roofline CLI、预处理时间、输入重排时间、输出恢复时间和静态 block
  指标。

CPU 只生成稳定排序顺序。CSR、事件和状态的大张量变换保持在原 tensor
device；CUDA 输入会直接使用 CUDA tensor 操作。

## 环境与工作负载

```text
environment: micromamba ml-py312
PyTorch: 2.10.0
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
dataset: mice_column_v1
neurons: 4,166
edges: 726,404
batch: 1
timesteps: 128
input event rate: 0.02
kernel: spike-block, block-hash disabled
timing: 10 warmup, 30 repeat, CUDA Event median
```

## 第一轮性能矩阵

不同方案在独立进程运行，因此小于约 3% 的差别应视为噪声。

| 策略 | W | preprocess ms | kernel ms | 相对 identity |
|---|---:|---:|---:|---:|
| identity | 256 | 2.42 | 7.169 | baseline |
| local cost bucket | 1024 | 49.33 | 7.206 | +0.5% |
| global cost bucket | 256 | 39.12 | 7.539 | +5.2% |
| local cost→similarity | 256 | 79.56 | 7.596 | +6.0% |
| local cost bucket | 256 | 25.34 | 8.826 | +23.1% |
| global similarity | 256 | 129.49 | 9.085 | +26.7% |
| global cost similar | 256 | 46.02 | 9.864 | +37.6% |
| global cost balanced | 256 | 34.74 | 10.024 | +39.8% |
| local cost→similarity | 1024 | 74.70 | 10.568 | +47.4% |
| global cost→similarity | 256 | 81.25 | 10.684 | +49.0% |

没有方案达到保留标准中的 kernel 时间下降 5%。最接近的
`local_cost_bucket, W=1024` 与 identity 基本持平。

## BlockTask 动态统计

instrumented block-stats 使用 5 warmup、20 repeat。延迟只用于同一
instrumented build 内比较。

| 指标 | identity | local cost bucket W=1024 | global similarity |
|---|---:|---:|---:|
| instrumented kernel ms | 6.040 | 9.644 | 9.965 |
| BlockTask count | 16,410 | 14,339 | 13,964 |
| active rows | 327,483 | 327,483 | 327,483 |
| active runs | 90,792 | 67,019 | 75,973 |
| span utilization | 0.300 | 0.817 | 0.811 |
| dynamic post duplicate ratio | 0.269 | 0.384 | 0.343 |
| long segment tasks | 92,282 | 92,282 | 92,282 |

重排达到了结构目标：局部粗分桶将动态 post 重复率提高约 11.5
个百分点，BlockTask 数下降 12.6%，active run 数下降 26.2%。但极长行
segment 数完全不变，而且把长行集中后会形成更重的 block 尾部。

静态指标也显示了同样的取舍：

```text
identity:
  block edge P99 = 26,111.2
  edges / unique post = 2.600

local cost bucket, W=1024:
  block edge P99 = 38,152.7
  edges / unique post = 3.368
```

## Hash 组合

启用现有 selective block hash 后：

| 策略 | kernel ms |
|---|---:|
| identity | 7.339 |
| global similarity | 8.773 |
| local cost bucket W=1024 | 9.823 |

新增 post 重复率没有被当前只覆盖 64--96 个 medium-row edges 的选择性
hash 转化为端到端收益。

## 正确性与运行时重排开销

CPU permutation、CSR 语义和所有动态张量 round-trip 测试通过。CUDA
端到端测试验证了重排图经过 spike-block kernel 和逆重排后，dense
spikes、event spikes、膜电位和 PSC 与原始编号路径一致。

在 `local_cost_bucket, W=1024` 的一次带正确性运行中：

```text
spike mismatch rate vs PyTorch = 4.13e-5
v max absolute difference = 0.0408
psc max absolute difference = 3.00e-4
input permutation = 0.101 ms
output restoration = 0.077 ms
```

误差与 identity 的 PyTorch 对照相同，来自既有 CUDA atomic 累加顺序。

## 结论

MVP 证明物理重排路径正确，也证明该图上存在可利用的动态 post
重复率。但当前 kernel 的主要限制不是普通 BlockTask 的 run 数或 post
重复率；长行 segment 尾部仍然存在。默认策略应继续保持 identity。

后续若继续，应先让极长行避免聚集到相邻 32-neuron block，或对 long
segment queue 做负载均衡，再针对 `local_cost_bucket, W=1024` 设计覆盖
更广 fanout 范围的 reduce。仅继续调整排序 key 不值得。

# RTX 5090 FlyBrain 重测

## 环境与工作负载

```text
server: cccl2
environment: micromamba ml-py312
PyTorch: 2.11.0+cu128
GPU: NVIDIA GeForce RTX 5090, GPU 3
driver: 595.71.05
dataset: flybrain / FlyWire 783
neurons: 138,639
edges: 15,091,983
average fanout: 108.86
batch: 1
timesteps: 128
input event rate: 0.02
timing: 10 warmup, 30 repeat, CUDA Event median
```

GPU 0--2 当时有其他负载，所有正式结果固定使用空闲 GPU 3。

identity permutation 增加快速路径后，其一次性准备时间从 44.8 ms
降至 0.185 ms。以下其他策略的预处理时间包含 permutation、CSR 和动态
输入的实际物理变换。

## 完整策略矩阵

| 策略 | W | preprocess ms | kernel ms | I/O permutation ms | E2E ms |
|---|---:|---:|---:|---:|---:|
| identity | 256 | 0.185 | 9.467 | 0 | 9.467 |
| global cost bucket | 256 | 207.65 | 9.849 | 0.150 | 9.998 |
| local cost→similarity | 256 | 161.10 | 9.972 | 0.152 | 10.124 |
| local cost bucket | 256 | 191.12 | 10.038 | 0.155 | 10.193 |
| global similarity | 256 | 281.74 | 10.152 | 0.219 | 10.371 |
| local cost bucket | 1024 | 147.64 | 10.308 | 0.152 | 10.460 |
| global cost→similarity | 256 | 237.68 | 10.421 | 0.225 | 10.646 |
| global cost balanced | 256 | 146.05 | 10.628 | 0.198 | 10.825 |
| local cost→similarity | 1024 | 190.73 | 11.516 | 0.155 | 11.670 |
| global cost similar | 256 | 151.60 | 11.718 | 0.200 | 11.918 |

最接近 identity 的 global coarse bucket，kernel 慢 4.0%，计入动态 I/O
后慢 5.6%。所有策略均未达到 5% 加速保留标准。

Similarity 的静态局部性改善明显，但同时放大了 block 尾部：

```text
identity:
  block edge P99 = 7,117.7
  edges / unique post = 1.041

global similarity:
  block edge P99 = 18,498.4
  edges / unique post = 1.280

global cost -> similarity:
  block edge P99 = 18,498.4
  edges / unique post = 1.291
```

## 活动率与 batch 敏感性

| event rate | batch | 策略 | kernel ms | E2E ms |
|---:|---:|---|---:|---:|
| 0.005 | 1 | identity | 9.060 | 9.060 |
| 0.005 | 1 | local cost bucket | 9.123 | 9.277 |
| 0.005 | 1 | global similarity | 9.633 | 9.859 |
| 0.005 | 1 | global cost bucket | 9.732 | 9.887 |
| 0.100 | 1 | identity | 10.494 | 10.494 |
| 0.100 | 1 | global cost bucket | 10.955 | 11.106 |
| 0.100 | 1 | global similarity | 11.161 | 11.382 |
| 0.100 | 1 | local cost bucket | 11.174 | 11.322 |
| 0.020 | 4 | identity | 30.626 | 30.626 |
| 0.020 | 4 | global cost bucket | 30.651 | 31.085 |
| 0.020 | 4 | local cost bucket | 30.757 | 31.193 |
| 0.020 | 4 | global similarity | 31.267 | 31.959 |

低活动率时 local coarse bucket 的纯 kernel 仅慢 0.7%，batch 4 时 global
coarse bucket 的纯 kernel 基本持平；但计入 permutation 后仍没有端到端
收益。

## 动态 BlockTask 统计

| 指标 | identity | local cost bucket W=256 | global similarity |
|---|---:|---:|---:|
| BlockTask count | 552,879 | 549,057 | 515,942 |
| active rows | 4,746,549 | 4,746,549 | 4,746,549 |
| active edges | 333,775,945 | 333,775,945 | 333,775,945 |
| active runs | 3,492,891 | 3,240,265 | 3,152,946 |
| span utilization | 0.216 | 0.317 | 0.306 |
| dynamic post duplicate ratio | 0.0178 | 0.0287 | 0.1373 |
| long segment tasks | 445,860 | 445,860 | 445,860 |

global similarity 将动态重复率提高到 13.7%，但对应的
`edges / unique_post` 约为 1.16，仍低于计划中建议继续开发 reduce 的
1.3。长行 segment 数完全没有改变。

## Hash 与正确性

启用 selective block hash：

| 策略 | kernel ms | E2E ms |
|---|---:|---:|
| identity | 9.548 | 9.548 |
| local cost bucket W=256 | 10.128 | 10.277 |
| global similarity | 10.298 | 10.524 |

hash 没有将新增重复率转化为收益。

FlyBrain 在 128 steps 时，identity 自身也会因强递归网络放大 CUDA
atomic 与 PyTorch reduction 顺序差异而超出既有最终状态容差。使用
单步窗口验证 graph 和状态变换：

```text
identity: spike mismatch 0, v error 0, PSC max error 7.63e-5
local cost bucket: spike mismatch 0, v error 0, PSC max error 1.53e-4
global similarity: spike mismatch 0, v error 0, PSC max error 9.16e-5
```

三种方案均通过单步正确性检查。

## 5090 结论

FlyBrain 与 mice_column_v1 的结论一致：重排可以改善 active run 和 post
局部性，但当前性能由 block-edge 长尾及未改变的 long-segment workload
主导。RTX 5090 上也应保持 identity 默认值。

下一步应实验“极长行不集中、普通行才按 cost/similarity 重排”，或者
直接重新平衡 long-segment queue。当前数据不支持继续优化现有全局
similarity 或 selective hash。
