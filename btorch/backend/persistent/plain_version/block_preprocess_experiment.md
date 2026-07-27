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
