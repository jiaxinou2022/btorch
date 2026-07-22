# Persistent SNN Block 化 Kernel 下一版改进计划

## 1. 目标与范围

当前 block 版本主要完成了：

- 以 32 个 neuron 为一组生成 `spike_mask`；
- 多个 spike 共用一个 `BlockTask`；
- 减少任务入队与领取次数；
- 将极长行切分为独立 segment task。

但原计划中的关键数据路径尚未真正实现：

1. block 内活跃 spike 对应突触数据的连续整块读取；
2. 利用 CSR 相对位置高效解析连续 edge 区间；
3. 跨多个活跃 row 的真正 warp/block reduce；
4. 消除普通行逐 row 串行处理；
5. 根据任务规模进行稳定的负载均衡。


## 2. 本版保留与暂缓内容

### 2.1 保留

- 当前 persistent kernel 总体框架；
- 32-neuron block 与 `spike_mask`；
- 普通 block task 与长行 segment task 两类任务；
- 当前 CSR 数据结构；
- 当前 direct global atomic 路径，作为基线与 fallback；
- 当前 benchmark、正确性测试和 profiler 流程。

本版优先在主 kernel 内直接实现并展开关键逻辑，避免额外函数边界及不可控的寄存器、调用和编译行为。

## 3. 目标执行路径

当前路径近似为：

```text
spike block
    → 生成一个 BlockTask
    → 逐条活跃 row 处理
    → 每条 edge 一次 global atomic
```

目标路径为：

```text
spike block
    → 解析 spike_mask
    → 找出活跃 row 的连续区间
    → 利用 CSR 相对位置形成连续 edge run
    → warp 连续读取 active edge stream
    → shared-memory hash 聚合相同 post
    → 少量 global atomic 写回
```

长行继续独立执行：

```text
long row
    → segment task
    → warp cooperative direct traversal
    → global atomic
```

长行不进入普通 block 的连续读取和 hash 聚合路径。

# 4. 阶段一：补充统计与基线（已完成，可之后用来调用分析）


# 5. 阶段二：正确实现 active-run 连续读取

这是下一版最高优先级。

## 5.1 从 spike mask 识别连续活跃区间

例如：

```text
spike_mask: 0011100011000100
```

解析为：

```text
run 0: lane [2, 5)
run 1: lane [8, 10)
run 2: lane [13, 14)
```

由于相邻 CSR row 在 edge 数组中首尾相接，一个连续活跃 row run 可直接转换为：

```cpp
edge_begin = graph_indptr[block_start + first_lane];
edge_end   = graph_indptr[block_start + end_lane];
```

无需复制 edge，也无需逐 row 保存任务。

## 5.2 利用相对位置直接展平读取

在突触传播阶段，只需要 `(post, weight)`，不需要知道每条 edge 属于哪一个 pre neuron。因此每个 run 可直接展平：

```cpp
for (int edge = edge_begin + lane;
     edge < edge_end;
     edge += warpSize) {
    int post = graph_indices[edge];
    float weight = graph_weight[edge];
    // direct atomic 或进入 shared hash
}
```

优势：

- run 边界仅计算一次；
- run 内没有 row 定位和二分查找；
- warp 读取连续 `graph_indices` 和 `graph_weight`；
- 多条相邻活跃 row 形成统一 edge stream；
- 为后续 shared hash 提供稳定输入。

## 5.3 第一版只实现严格 active-run

只合并真正相邻的活跃 row，不跨越 inactive row。

例如：

```text
active lanes: 2, 3, 4, 7, 8
```

处理：

```text
[indptr[2], indptr[5])
[indptr[7], indptr[9])
```

暂时不直接读取：

```text
[indptr[2], indptr[9])
```

避免低发放率下大量读取 inactive row。

## 5.4 边界情况

必须覆盖：

- `spike_mask == 0`；
- 只有一个 active lane；
- 全 32 lane active；
- run 位于 lane 31；
- neuron 数不是 32 的整数倍；
- 普通行与长行处于同一个原始 block；
- 长行已被单独入队后，不能在普通 block 路径重复处理；
- 位运算不能出现 `1u << 32` 等未定义行为。

## 5.5 对照实验

固定调度和 atomic 策略，只比较读取方式：

```text
A. 当前 BlockTask 逐 row 处理
B. active-run 连续读取
```

关注：

- kernel 总时间；
- 中等 fanout 17～255；
- warp active lanes；
- branch divergence；
- graph edge load 吞吐；
- Long Scoreboard；
- active row 数与 run 数之比。

## 5.6 阶段验收

- 数值与原实现一致；
- 不遗漏、不重复任何 edge；
- 相邻活跃 row 只形成一次连续 traversal；
- 中等行不再全部逐 row 串行；
- run 统计与理论结果一致；
- 即使总体暂未加速，也不能出现明显的大范围退化。

# 6. 阶段三：处理离散 spike mask 与串行执行

active-run 适合连续发放，但原始 neuron 顺序下，mask 可能很离散。

## 6.1 保留极短行 lane-direct 路径

适用条件初始设为：

```text
degree <= 8 或 16
```

执行方式：

```text
一个 lane 对应一个 active neuron
每个 lane 串行遍历自己的短 row
```

阈值扫描：

```text
4, 8, 16, 32
```

## 6.2 连续 active-run 路径

适合：

- 至少存在长度大于 1 的连续 active run；
- active edge 数达到一定规模；
- run 合并可以明显减少 traversal 次数。

## 6.3 离散 active rows 的 packed edge stream

当 mask 类似：

```text
0100010001000010
```

run 数接近 active row 数，逐 run 执行仍然接近逐 row 串行。

此时在 warp 内构造紧凑 active row 描述：

```text
active_pre[0..K)
row_begin[0..K)
row_degree[0..K)
prefix[0..K]
```

逻辑 edge 编号：

```text
logical_edge ∈ [0, total_active_edges)
```

warp 统一循环处理：

```cpp
for (int logical_edge = lane;
     logical_edge < total_active_edges;
     logical_edge += warpSize) {
    // 根据 prefix 找到所属 active row
    // 转换到实际 CSR edge
}
```

第一版使用最多 32 项的小数组和简单定位，不加入复杂通用二分结构。

## 6.4 初始路径选择规则

```text
if 所有 active row 都很短:
    lane-direct
else if active_run_count * 2 <= active_rows:
    active-run traversal
else:
    packed active rows
```

后续根据统计调整。

## 6.5 阶段验收

- 重点观察 fanout 17～255；
- warp issue efficiency 改善；
- eligible warps 改善；
- 离散 mask 不再造成大量短 traversal；
- lane-direct、active-run、packed 三条路径占比合理；
- 分支和局部数组没有造成明显寄存器膨胀。

# 7. 阶段四：shared-memory hash table 聚合

连续或 packed active edge stream 稳定后，再加入真正跨 row 聚合。

## 7.1 目标

当前：

```text
每条 active edge → 一次 global atomicAdd
```

目标：

```text
多个指向同一 post 的 active edge
    → shared hash 合并
    → 每个 unique post 一次 global atomicAdd
```

成功判据：

```text
hash_flush_count < active_edges
```

## 7.2 第一版结构

每个处理 warp 使用独立 hash 区域：

```cpp
__shared__ int hash_keys[WARPS_PER_BLOCK][HASH_SIZE];
__shared__ float hash_values[WARPS_PER_BLOCK][HASH_SIZE];
```

扫描：

```text
HASH_SIZE = 64, 128, 256
```

优先从 128 开始。

## 7.3 初始化

```cpp
for (int slot = lane; slot < HASH_SIZE; slot += warpSize) {
    hash_keys[warp_id][slot] = EMPTY;
    hash_values[warp_id][slot] = 0.0f;
}
__syncwarp();
```

第一版直接清表，不立即实现 epoch/tag。

## 7.4 插入

```cpp
slot = hash(post) & (HASH_SIZE - 1);

for (int probe = 0; probe < MAX_PROBE; ++probe) {
    int old = atomicCAS(&hash_keys[warp_id][slot], EMPTY, post);

    if (old == EMPTY || old == post) {
        atomicAdd(&hash_values[warp_id][slot], weight);
        success = true;
        break;
    }

    slot = (slot + 1) & (HASH_SIZE - 1);
}
```

若超过 `MAX_PROBE`：

```cpp
atomicAdd(global_psc + post, weight);
```

扫描：

```text
MAX_PROBE = 4, 8, 16
```

## 7.5 Flush

```cpp
__syncwarp();

for (int slot = lane; slot < HASH_SIZE; slot += warpSize) {
    int key = hash_keys[warp_id][slot];
    if (key != EMPTY) {
        atomicAdd(global_psc + key,
                  hash_values[warp_id][slot]);
    }
}
```

## 7.6 Hash 启用条件

初始启发式：

```text
active_rows >= 2
active_edges >= 64
active_edges <= hash_edge_limit
不包含长行
```

扫描：

```text
active_edges threshold = 32, 64, 96, 128
```

不满足则走 direct atomic。

## 7.7 Hash microbenchmark

在接入完整 kernel 前单独测试：

```text
active_edges: 32, 64, 128, 256, 512
duplicate ratio: 0%, 25%, 50%, 75%
post distribution: 均匀随机、少量热点、分块热点、Zipf/长尾
```

比较：

```text
direct global atomic
match_any_sync
shared hash
```

扫描 `HASH_SIZE` 和 `MAX_PROBE`，确定盈亏边界后再设置主 kernel 启用条件。

## 7.8 阶段验收

- 数值正确；
- global atomic 数真实下降；
- 高重复率场景明确加速；
- 低重复率场景正确 fallback；
- hash overflow 比例可控；
- shared-memory 使用不会导致 occupancy 大幅下降；
- 寄存器和 local memory spill 可控。

# 8. 阶段五：普通任务与长行任务负载均衡

只有新数据路径稳定后再优化调度器。

## 8.1 保持双队列

```text
normal block queue
long segment queue
```

普通 block 使用 lane-direct、active-run、packed rows 和 optional hash；长行 segment 使用 warp cooperative direct traversal。

## 8.2 长行优先级

尝试：

```text
每次优先检查 long queue
long queue 为空再取 normal task
```

或：

```text
每处理 K 个 normal task 检查一次 long queue
```

扫描：

```text
K = 1, 2, 4, 8
```

## 8.3 长行 segment size

扫描：

```text
segment_size = 128, 256, 512, 1024
```

观察 task 数、counter 成本、最慢 warp、kernel 尾部利用率及 atomic 吞吐。

## 8.4 小批量任务领取

尝试：

```cpp
base = atomicAdd(counter, reserve_size);
```

扫描：

```text
reserve_size = 1, 2, 4
```

要求：

- 领取后仍逐个处理；
- 不将多个异质任务绑定成不可拆超级任务；
- 首轮不测试 8 或 16；
- 对比 counter 减少与负载不均增加。

## 8.5 重任务 fallback

若普通 block 的 `active_edges` 过大：

```text
active_edges > block_task_limit
    → 禁用 hash
    → 使用 direct packed/run traversal
```

暂不动态拆分新子任务。

## 8.6 阶段验收

- kernel 尾部空闲减少；
- 最慢任务时间下降；
- 长行不阻塞普通 block；
- reserve 不造成明显负载不均；
- 不同 event rate 与 fanout 分布下表现稳定。

# 9. 主 kernel 组织原则

本版不抽取 device function，关键路径直接写在主 kernel 或现有主循环中。

建议按代码区段组织：

```text
1. fetch task
2. 判断 long / normal
3. normal task 解析 spike_mask
4. 计算 active_rows、degree、active_edges、run_count
5. 选择 lane-direct / active-run / packed
6. 选择 direct atomic / shared hash
7. flush
8. 获取下一任务
```

## 9.1 控制寄存器压力

- 路径专用变量限制作用域；
- 不同时保留 active-run 与 packed 路径的全部局部数组；
- shared hash 变量仅在启用路径中使用；
- 避免复杂结构体承载 metadata；
- 只保留少量标量；
- 每次修改后检查 registers/thread、spill、active blocks/SM 和 occupancy。

## 9.2 逐步增加路径

```text
版本 1：tiny direct + active-run direct + long segment
版本 2：增加 packed fallback
版本 3：增加 selective hash
版本 4：增加 task reserve 与调度优化
```

每版保留独立 benchmark。

# 10. 实验矩阵

## 10.1 数据维度

- event rate：0.005、0.02、0.1；
- batch：1、4、典型实际 batch；
- fanout：8、16、32、64、128、256、512、1024、2050；
- 分布：固定 fanout、长尾 fanout、真实网络、高 post overlap、低 post overlap。

## 10.2 版本对照

```text
V0：原始 persistent
V1：当前 BlockTask 版本
V2：active-run direct
V3：active-run + packed fallback
V4：V3 + selective shared hash
V5：V4 + 调度优化
```

## 10.3 指标

性能：

- 总推理时间；
- kernel 时间；
- timestep 时间；
- speedup；
- 方差。

Profiler：

- Long Scoreboard；
- memory throughput；
- L1/L2 hit rate；
- atomic throughput；
- warp issue efficiency；
- eligible warps；
- branch efficiency；
- occupancy；
- registers/thread；
- shared memory/block。

算法统计：

- active rows/task；
- run count/task；
- active edges/task；
- span utilization；
- hash merge ratio；
- hash overflow ratio；
- global atomic reduction；
- 各执行路径比例。

# 11. 失败判据与回退原则

## 11.1 Active-run 无收益

若：

```text
active_run_count ≈ active_rows
```

说明现有 neuron 顺序下活跃 spike 很少连续。保留正确实现，主要依赖 packed fallback，并把 neuron reorder 延后到下一阶段。

## 11.2 Packed traversal 负优化

可能原因：prefix 构造成本、physical load 离散、active edges 太少或 row 定位过重。

处理：只对 active rows/edges 达到阈值的任务启用，小任务回退 lane-direct 或逐 run。

## 11.3 Hash 负优化

可能原因：duplicate ratio 低、清表与 flush 成本、shared atomic 冲突、collision、occupancy 下降或任务太小。

处理：提高启用阈值、调整 hash size、限制 probe、overflow 直接 global atomic。

## 11.4 调度优化负优化

可能原因：reserve 太大、重任务被单 warp 连续领取、队列优先级不合理、segment 粒度不合适。

处理：回退 reserve=1，重新扫描 segment size，并保留双队列。

# 12. 里程碑

## Milestone 1：正确的 block 连续读取

```text
spike_mask
    → active runs
    → CSR 相对边界
    → 连续 edge traversal
```

验收：数值正确，相邻活跃 row 一次读取，不再全部逐 row 串行。

## Milestone 2：消除离散 mask 的串行瓶颈

```text
离散 active rows
    → packed logical edge stream
    → warp 并行消费
```

验收：中等 fanout 下 lane 利用率改善，分散 mask 不再造成大量短 traversal。

## Milestone 3：真正的 block-private reduce

```text
active edge stream
    → shared hash
    → unique post flush
```

验收：global atomic 数真实下降，高重复场景加速，低重复场景 fallback。

## Milestone 4：稳定负载均衡

```text
normal block + long segment
    → 合理优先级
    → 小批量领取
    → segment size 调优
```

验收：kernel 尾部空闲降低，长行不拖累普通 block。

## Milestone 5：决定是否进入物理重排

根据统计决定是否需要：

```text
长行统一移到后部
normal neuron 物理重排
CSR row/column/state 同步 permutation
post 相似性聚类
```

触发条件：active runs 过碎、block 内 degree 差异大、post 重复率低或 hash 收益受 neuron 排列限制。

# 13. 推荐实施顺序

```text
P0：补充 debug 统计
P1：active-run 连续读取
P2：扫描 tiny lane-direct 阈值
P3：packed active-row fallback
P4：shared hash microbenchmark
P5：选择性接入 shared hash
P6：调 long segment size
P7：小批量任务领取
P8：分析是否需要 neuron 物理重排
```

本版最重要的是 P1、P3 和 P5：

- P1 恢复真正的 block 连续读取；
- P3 消除当前普通行逐 row 串行；
- P5 实现真正减少 global atomic 的 warp/block reduce。

在这些路径得到正确实现和可靠统计之前，不继续扩大预处理复杂度。
