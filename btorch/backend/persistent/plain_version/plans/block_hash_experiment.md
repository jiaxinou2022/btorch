# Persistent Block Hash FlyBrain 实验

## 实现

在 Block V4 的 edge-budget consumer 中加入编译期可配置的
warp-private shared-memory hash：

```text
BTORCH_BLOCK_HASH_AGGREGATION = 128 / 256 / 512
BTORCH_BLOCK_HASH_CAPACITY    = 128 / 256 / 512
BTORCH_BLOCK_HASH_MAX_PROBE   = 4 / 8 / 16
BTORCH_BLOCK_HASH_MIN_EDGES   = 0 / 64 / 128 / 192 / 256
BTORCH_BLOCK_HASH_USED_SLOTS  = 0 / 1
```

每个 32-edge 批次先用 `atomicCAS` 建立 key，warp 同步后再对 shared
value 执行 `atomicAdd`。探测失败的 edge 回退到 global atomic；窗口结束后
occupied slot 各写回一次。hash 路径不使用 `match_any`。

基础版本每个窗口初始化和扫描完整 hash。used-slot 版本在 kernel 开始时
初始化一次 key；每个 32-edge 批次用 ballot/prefix rank 记录本批新占用
slot，不使用 shared counter atomic。窗口结束只 flush 并清理实际使用的
slot。

当 aggregation 为 128 时，一个 256-edge logical task 分两个窗口；256
覆盖一个 logical task；512 将相邻 logical segment 合成一个 warp chunk。
512 因此同时改变聚合范围和调度粒度。

benchmark 新增：

```text
--hash-aggregation
--hash-capacity
--hash-max-probes
--hash-min-edges
--hash-used-slots
--wait-idle-samples
```

stats build 直接记录 hash input edges、flush atomics、fallback atomics 和
probe attempts。离线统计按 32/64/128/256/512/full-block 窗口重建精确的
unique-post 理论上限。

opt-in `--block-hash` 的默认值已收敛为实验最佳：

```text
aggregation=512, capacity=512, probes=4, min_edges=256, used_slots=on
```

## 环境

```text
server: cccl2
GPU: NVIDIA GeForce RTX 5090, GPU 3
environment: micromamba ml-py312
dataset: FlyBrain / FlyWire 783
neurons: 138,639
edges: 15,091,983
reorder: global_similarity
block edge budget: 256
long segment size: 512
event rate: 0.02
batch: 1
```

计数和理论上限使用 8 个代表性时间步；该样本包含 22,388,482 条普通
BlockTask edge。正式计时使用 128 时间步、10 warmup、50 repeat。

## 理论聚合上限

| scale | ideal atomic reduction | P90 edges/unique | edge coverage ratio >= 1.1 |
|---:|---:|---:|---:|
| 32 | 0.059% | 1.000 | 0.265% |
| 64 | 0.187% | 1.000 | 0.606% |
| 128 | 1.086% | 1.032 | 3.705% |
| 256 | 3.442% | 1.108 | 12.157% |
| 512 | 6.720% | 1.177 | 25.205% |
| full block | 13.029% | 1.268 | 49.322% |

256-edge 聚合低于计划的 5% 停止线。full-block 明显高于 256，说明重复
主要跨 logical task，但直接扩大到完整 block 会重新引入 V4 已消除的长尾。

## 实际 hash 压缩

| aggregation | capacity | probes | min edges | global atomic reduction | fallback / hash edges | average probes |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 256 | 8 | 128 | 1.028% | 0.521% | 1.448 |
| 256 | 256 | 8 | 128 | 3.379% | 9.850% | 2.486 |
| 256 | 512 | 8 | 128 | 3.412% | 0.394% | 1.413 |
| 256 | 512 | 4 | 128 | 3.401% | 2.342% | 1.352 |
| 256 | 512 | 4 | 256 | 3.107% | 2.502% | 1.364 |
| 512 | 512 | 8 | 128 | 6.621% | 7.188% | 2.246 |
| 512 | 512 | 4 | 128 | 6.470% | 11.871% | 1.760 |
| 512 | 512 | 4 | 256 | 6.310% | 12.640% | 1.796 |

capacity 512 将 256-edge 的 fallback 从 9.85% 降到 0.39%，并把平均
probe 从 2.49 降到 1.41。将 probe limit 从 8 降到 4 几乎不损失最终
atomic reduction，同时进一步减少 probe 工作。

## 延迟扫描

服务器实验期间四张 RTX 5090 都有其他训练进程。GPU 3 的后台任务在
0% 和约 22% utilization 之间周期性切换。外部瞬时空闲门槛仍会在
Python 数据准备后失效，因此 benchmark 增加了计时点前的连续空闲检测。
连续 3 秒空闲在五分钟内未出现，最终使用计时点前连续 1 秒、5 个样本
均不高于 2% 的门槛。相同配置重复两次，CUDA Event 各取 50 次中位数。

| aggregation | capacity | probes | min edges | used slots | kernel ms | E2E ms |
|---:|---:|---:|---:|:---:|---:|---:|
| off | - | - | - | - | 11.894 | 12.121 |
| 128 | 256 | 8 | 128 | off | 12.029 | 12.252 |
| 256 | 512 | 4 | 128 | off | 10.953 | 11.177 |
| 256 | 512 | 4 | 256 | off | 10.931 | 11.156 |
| 512 | 512 | 8 | 128 | off | 10.500 | 10.720 |
| 512 | 512 | 4 | 128 | off | 10.402 | 10.625 |
| 512 | 512 | 4 | 256 | off | 10.245 | 10.464 |
| 512 | 512 | 4 | 256 | on | 10.167 | 10.391 |

最终 used-slot 配置相对同轮 direct-atomic V4：

```text
kernel: 14.5% faster
E2E:    14.3% faster
```

used-slot 相对全表初始化/flush 仅快约 0.8%，属于小收益；主要收益来自
512-edge 调度/聚合和减少 6.31% global atomic。128-edge hash 因理论重复
仅 1.09%，实际比 baseline 慢 1.1%。

### 分离调度和 hash 收益

aggregation=512 会把两个 256-edge logical task 合成一个 warp chunk。使用
`block-edge-budget=512`、hash off 作为调度匹配对照：

| reorder | budget 256 direct | budget 512 direct | 512 hash |
|---|---:|---:|---:|
| global similarity | 11.894 ms | 11.352 ms | 10.167 ms |
| identity | 10.189 ms | 9.743 ms | 9.369 ms |

对 global similarity，约 4.6% 来自 task 合并；hash 相对匹配的
512-edge direct 路径再快 10.4%。因此最终收益不是单纯由调度变化产生。

identity 的 512-edge 理论/实际 atomic reduction 仅为 0.85%/0.79%，hash
相对匹配 direct 路径仍快 3.8%，可能还受 flush 后写回顺序和 global atomic
局部性影响。global similarity 虽有更强压缩，但重排后的静态重 block 更重，
最终绝对延迟仍比 identity 高。本数据点绝对最佳为 identity + hash 的
9.369 ms，无额外 permutation/restore 成本。

### 活动率和 batch 敏感性

以下比较保持 global similarity，以隔离 hash 变化：

| event rate | batch | direct kernel | hash kernel | kernel gain | E2E gain |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 1 | 9.548 ms | 8.530 ms | 10.7% | 10.7% |
| 0.020 | 1 | 11.894 ms | 10.167 ms | 14.5% | 14.3% |
| 0.100 | 1 | 13.499 ms | 11.564 ms | 14.3% | 14.1% |
| 0.020 | 4 | 37.639 ms | 33.039 ms | 12.2% | 12.0% |

与上一版 V4 不同，hash/512-edge chunk 在低活动率和 batch=4 仍保持正收益。

## 正确性和停止条件

- FlyBrain 单步 PyTorch 对照通过，spike mismatch 为 0；
- PSC 最大绝对误差为 1.22e-4，在既有容差内；
- aggregation 128/256/512 的 synthetic CUDA 对照均通过；
- 9 个 spike-block CUDA 回归测试通过；
- 7 个统计单元测试通过。

256-edge 实际 atomic reduction 只有约 3.4%，但 512-edge 聚合达到 6.31%
并获得稳定正收益，因此实施了 used-slot 优化。full-block 理论上限虽为
13.03%，但继续扩大 warp chunk 会重新引入长尾；当前不实施 epoch/tag 或
CTA-wide hash。
