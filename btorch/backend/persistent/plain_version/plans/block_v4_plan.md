# Persistent Block V4 实施计划

## 1. 总体目标

当前 BlockTask 固定由连续 32 个 neuron 的 spike mask 表示：

```text
task = {
    batch,
    neuron_block_start,
    spike_mask
}
```

一个 task 的实际成本却由其中发放 neuron 的总 fanout 决定：

[
E_{\text{active}}
=================

\sum_{i\in\text{spike mask}}\deg(i)
]

相似性或成本聚合后，少数 task 的 `active_edges` 可能非常大，形成时间步末尾的重任务。

下一版保留固定 32-neuron block，不修改物理重排定义，但允许一个 spike mask 根据 active-edge 数量被拆成多个逻辑任务：

```text
一个 32-neuron spike block
    ↓ active-edge prefix
一个或多个 edge-budget task
```

每个逻辑 task 只负责：

[
[\text{logical edge begin},\text{logical edge end})
]

从而限制单 warp 最长工作量。

同时，针对相似性重排产生的重复 post，在逻辑 task 内做自适应 reduce，将多条写向相同 post 的边合并为一次 global atomic。

极长行仍走独立 long-segment 路径，只调整 segment size，不与普通 BlockTask 混合。

---

# 2. 修改一：UPDATE 阶段累计 active edges 并按预算打包

## 2.1 核心思路

UPDATE 时，一个 warp仍然处理连续 32 个 neuron。

每个 lane计算：

```text
fired
degree
ordinary = fired && degree < extreme_threshold
```

然后对普通发放行的 degree 做 warp prefix sum：

```text
lane_edge_begin
lane_edge_end
total_active_edges
```

设置一个可调预算：

```cpp
BLOCK_EDGE_BUDGET = 256 / 512 / 1024
```

若：

```text
total_active_edges <= BLOCK_EDGE_BUDGET
```

则生成一个普通 BlockTask。

若超过预算，则生成：

[
\left\lceil
\frac{total_active_edges}
{BLOCK_EDGE_BUDGET}
\right\rceil
]

个逻辑子任务。

每个子任务共享同一个：

```text
batch
neuron_block_start
spike_mask
```

但额外保存：

```text
logical_edge_begin
logical_edge_end
```

这样不需要改变物理 neuron block，也不需要在预处理时动态重分块。

---

## 2.2 为什么在 UPDATE 阶段切分

UPDATE 阶段已经拥有：

* fired mask；
* 每个 lane 对应的 neuron；
* 每个 neuron 的 degree；
* 当前时间步真实 active row。

所以这里能够依据**动态 active-edge 总量**切分，而不是依靠静态 block fanout估计。

这比预处理阶段按静态总fanout切块更准确：

```text
静态高fanout但本步未发放
→ 不产生任务成本

静态普通但本步多行同时发放
→ 可动态切分
```

---

## 2.3 数据结构

建议将普通 task descriptor 扩展为：

```cpp
struct BlockTask {
    int batch;
    int block_start;
    uint32_t spike_mask;

    uint32_t logical_begin;
    uint32_t logical_end;
};
```

但为减少内存事务，可以继续使用 SoA：

```cpp
task_batch[]
task_block_start[]
task_spike_mask[]
task_logical_begin[]
task_logical_end[]
```

若 `BLOCK_EDGE_BUDGET <= 65535`，logical offset 可以考虑压缩为16 bit，但第一版先使用32 bit，避免增加解码复杂度。

---

## 2.4 UPDATE 阶段伪代码

```cpp
constexpr int kWarpSize = 32;
constexpr int kExtremeThreshold = 256;
constexpr int kBlockEdgeBudget = 512;

lane = threadIdx.x & 31;
warp_mask = __activemask();

neuron = warp_neuron_base + lane;

fired = update_neuron_and_test_spike(neuron);
degree = graph_indptr[neuron + 1] - graph_indptr[neuron];

ordinary_spike =
    fired && degree > 0 && degree < kExtremeThreshold;

extreme_spike =
    fired && degree >= kExtremeThreshold;

// ----------------------------------------
// A. 极长行仍进入 long queue
// ----------------------------------------

if (extreme_spike) {
    segment_count =
        ceil_div(degree, LONG_SEGMENT_SIZE);

    first =
        atomicAdd(long_task_count, segment_count);

    for segment = 0 .. segment_count - 1:
        long_task_batch[first + segment] = batch;
        long_task_row[first + segment] = neuron;
        long_task_edge_begin[first + segment] =
            row_begin + segment * LONG_SEGMENT_SIZE;
        long_task_edge_end[first + segment] =
            min(row_end,
                row_begin
                + (segment + 1) * LONG_SEGMENT_SIZE);
}

// ----------------------------------------
// B. 普通行建立动态 logical edge stream
// ----------------------------------------

row_edges = ordinary_spike ? degree : 0;

// warp inclusive scan
inclusive_end = warp_prefix_sum(row_edges);
exclusive_begin = inclusive_end - row_edges;

total_active_edges =
    __shfl_sync(warp_mask, inclusive_end, 31);

spike_mask =
    __ballot_sync(warp_mask, ordinary_spike);

if (lane == 0 && total_active_edges > 0) {
    task_count =
        ceil_div(
            total_active_edges,
            kBlockEdgeBudget);

    first_task =
        atomicAdd(block_task_count, task_count);

    for segment = 0 .. task_count - 1 {
        logical_begin =
            segment * kBlockEdgeBudget;

        logical_end =
            min(total_active_edges,
                logical_begin + kBlockEdgeBudget);

        task_batch[first_task + segment] =
            batch;

        task_block_start[first_task + segment] =
            warp_neuron_base;

        task_spike_mask[first_task + segment] =
            spike_mask;

        task_logical_begin[first_task + segment] =
            logical_begin;

        task_logical_end[first_task + segment] =
            logical_end;
    }
}
```

---

## 2.5 UPDATE 阶段本地清空并继续 spike

用户提出的“超过临界值后直接打包进入队列，同时本地清空继续 spike”，可以有两种实现。

### 方案 A：先累计完整 warp，再切多个逻辑任务

即上面的实现。

优点：

* 只做一次warp prefix；
* spike mask只生成一次；
* 实现简单；
* 不需要在warp扫描32个neuron过程中维护复杂状态。

缺点：

* lane 0可能串行写多个task descriptor；
* 超重block仍需生成多个queue entry。

建议作为第一版。

### 方案 B：逐lane流式累计，达到预算立即flush

概念上：

```text
扫描 lane 0→31
累计当前packet的row和edge数
超过预算：
    flush当前packet
    清空本地packet
    当前row进入下一packet
```

但CUDA warp不是按lane串行执行。实现它需要：

* prefix/boundary检测；
* 一个warp可能生成多个mask；
* 某条row可能跨两个packet；
* 维护每个packet的row mask和partial-row offset。

复杂度明显更高。

因此第一版不建议真正“边扫描边清空”，而应采用：

> 一次生成完整spike mask和prefix，再根据logical edge range切分。

它在执行效果上等价于按预算分包，同时保持实现简洁。

---

# 3. 普通 BlockTask 的消费

## 3.1 构建 row prefix

warp领取task后，读取：

```text
block_start
spike_mask
logical_begin
logical_end
```

每个lane对应一条row，计算：

```text
row_begin
row_end
row_degree
```

若lane未发放，则 `row_degree=0`。

随后做warp prefix，得到每条row在完整逻辑edge stream中的范围：

```text
row logical range:
[prefix_begin[lane], prefix_end[lane])
```

当前task只处理与：

```text
[logical_begin, logical_end)
```

相交的部分。

---

## 3.2 映射 logical edge 到实际 CSR edge

第一版可以沿用当前 prefix-pack 的二分定位：

```cpp
owner_lane =
    upper_bound(prefix_end, logical_edge);

offset_in_row =
    logical_edge - prefix_begin[owner_lane];

physical_edge =
    graph_indptr[block_start + owner_lane]
    + offset_in_row;
```

由于每个task的logical edge数被限制在256/512/1024内，单个warp尾部有了明确上界。

后续再优化二分成本。

---

## 3.3 消费伪代码

```cpp
task = claim_block_task();

batch = task_batch[task];
block_start = task_block_start[task];
spike_mask = task_spike_mask[task];
logical_begin = task_logical_begin[task];
logical_end = task_logical_end[task];

lane = threadIdx.x & 31;

row = block_start + lane;

active =
    ((spike_mask >> lane) & 1) != 0;

row_begin =
    active ? graph_indptr[row] : 0;

row_degree =
    active
        ? graph_indptr[row + 1] - row_begin
        : 0;

prefix_end =
    warp_prefix_sum(row_degree);

prefix_begin =
    prefix_end - row_degree;

for (logical_base = logical_begin;
     logical_base < logical_end;
     logical_base += 32)
{
    logical_edge =
        logical_base + lane;

    valid =
        logical_edge < logical_end;

    if (valid) {
        owner =
            warp_upper_bound(
                prefix_end,
                logical_edge);

        owner_prefix_begin =
            __shfl_sync(
                FULL_MASK,
                prefix_begin,
                owner);

        owner_row_begin =
            __shfl_sync(
                FULL_MASK,
                row_begin,
                owner);

        physical_edge =
            owner_row_begin
            + logical_edge
            - owner_prefix_begin;

        post =
            graph_indices[physical_edge];

        weight =
            graph_weight[physical_edge];
    }

    process_or_reduce(post, weight, valid);
}
```

---

# 4. 修改二：加入 block/warp 内 reduce

## 4.1 目标

相似性重排提高了同一普通BlockTask中写向相同post的概率，但目前仍然是一条edge一次global atomic。

reduce的目标是：

```text
(post=A, w0)
(post=A, w1)
(post=A, w2)
```

变成：

```text
(post=A, w0+w1+w2)
→ 一次global atomic
```

当前FlyBrain的整体 `edges / unique post≈1.16`，不足以支持对所有task启用重型hash，因此必须采用：

> **轻量、自适应、分级启用的reduce。**

---

## 4.2 第一版优先：32-edge tile reduce

每次warp处理最多32条edge，先在这一轮内部合并相同post。

它不跨多个tile维护hash，因此没有：

* 每task固定128槽初始化；
* 全表清空；
* 全表flush；
* 大型shared状态。

### 基础伪代码

```cpp
valid_mask = __ballot_sync(
    FULL_MASK,
    valid);

peer_mask =
    __match_any_sync(
        valid_mask,
        post);

leader =
    __ffs(peer_mask) - 1;

sum = weight;

// peer reduction
for (offset = 16; offset > 0; offset >>= 1) {
    other_post =
        __shfl_down_sync(
            peer_mask,
            post,
            offset);

    other_weight =
        __shfl_down_sync(
            peer_mask,
            sum,
            offset);

    if (lane + offset < 32
        && other_post == post)
    {
        sum += other_weight;
    }
}

if (valid && lane == leader) {
    atomicAdd(
        &psc[batch * n_neuron + post],
        sum);
}
```

但上述 `shfl_down` 与任意peer mask组合需要仔细实现，不能直接假设相同key lane连续。

更可靠的第一版可由leader遍历peer：

```cpp
peer_mask =
    __match_any_sync(valid_mask, post);

leader =
    __ffs(peer_mask) - 1;

if (lane == leader) {
    float sum = 0;

    unsigned remaining = peer_mask;

    while (remaining != 0) {
        int src =
            __ffs(remaining) - 1;

        sum +=
            __shfl_sync(
                peer_mask,
                weight,
                src);

        remaining &=
            remaining - 1;
    }

    atomicAdd(
        &psc[batch * n_neuron + post],
        sum);
}
```

这一版更容易保证正确，但重复组很大时leader循环会串行。适合作为功能与收益验证。

---

## 4.3 避免所有tile无条件执行 reduce

此前 `match_any_sync` 已表现出较高开销，因此需要启用条件。

### 条件 A：预处理相似性标签

如果当前32-neuron block属于高相似性block，才启用tile reduce。

预处理保存：

```cpp
block_reduce_hint[block_id]
```

例如根据静态：

[
\frac{\text{block edges}}
{\text{block unique posts}}
]

决定。

### 条件 B：运行时轻量范围判断

先计算tile内：

```text
min_post
max_post
```

若：

```text
max_post - min_post
```

很大，重复概率通常较低，直接atomic。

若post范围较集中，再执行match/reduce。

### 条件 C：只对大task启用

若：

```text
logical_end - logical_begin < 64
```

固定reduce开销可能不值得。

可测试：

```cpp
enable_reduce =
    task_edges >= REDUCE_MIN_EDGES
    && block_reduce_hint[block_id];
```

---

## 4.4 第二版候选：warp-private小hash

若tile reduce证明global atomic减少有收益，但跨tile重复很多，可以再做小hash。

建议：

```text
hash size = 32 或64
warp-private
epoch/tag清空
只对高重复task启用
```

不要直接恢复原先128槽固定初始化方案。

伪代码：

```cpp
if (enable_hash) {
    begin_hash_epoch();

    for each edge tile:
        for each valid edge:
            if (!hash_insert_or_accumulate(post, weight)):
                atomicAdd(global_psc[post], weight);

    flush_hash_to_global();
}
else {
    direct_or_tile_reduce();
}
```

---

## 4.5 必须新增的统计

每种reduce路径都记录：

```text
input edges
unique posts
global atomics emitted
tile-reduced atomics
hash fallback atomics
reduce-enabled task count
reduce-enabled edge count
```

核心指标：

[
R_{\text{atomic}}
=================

\frac{\text{global atomics emitted}}
{\text{input active edges}}
]

只有这个比例显著下降，才能说重复率真正转化成执行收益。

---

# 5. 修改三：极长行 segment size 扫描

## 5.1 当前问题

极长行segment任务数量在不同重排下完全不变，因为重排没有改变：

* 极长行degree；
* fired次数；
* segment size；
* 每条边global atomic。

所以long segment是旧瓶颈，不是普通block切分可以消除的。

segment size控制两类成本的权衡。

### segment太大

* 单warp处理轮数多；
* 时间步末尾单task尾部重；
* 任务数量不足；
* atomic冲突严重task更难被平滑。

### segment太小

* task数量增加；
* queue descriptor增加；
* UPDATE阶段segment生成循环变长；
* work-counter atomic增加；
* metadata读取和调度开销增加。

---

## 5.2 测试范围

建议扫描：

```text
128
256
512
1024
2048
```

其中：

| segment size | 每warp最大轮数 |
| -----------: | --------: |
|          128 |         4 |
|          256 |         8 |
|          512 |        16 |
|         1024 |        32 |
|         2048 |        64 |

FlyBrain在5090上任务量较充分，最优值可能落在256–1024之间。

---

## 5.3 参数化实现

```cpp
template<int LongSegmentSize>
__global__ void persistent_kernel(...)
{
    ...
}
```

或者运行时参数：

```cpp
int long_segment_size;
```

模板版本通常更利于编译器优化，但会生成多个kernel variant。

建议第一轮模板化：

```text
persistent_block_seg128
persistent_block_seg256
persistent_block_seg512
persistent_block_seg1024
persistent_block_seg2048
```

---

## 5.4 极长行任务生成伪代码

```cpp
if (fired && degree >= kExtremeThreshold) {
    int segment_count =
        ceil_div(
            degree,
            LongSegmentSize);

    int first_task =
        atomicAdd(
            long_task_count,
            segment_count);

    for (int segment = 0;
         segment < segment_count;
         ++segment)
    {
        int begin =
            row_begin
            + segment * LongSegmentSize;

        int end =
            min(row_end,
                begin + LongSegmentSize);

        long_task_batch[first_task + segment] =
            batch;

        long_task_edge_begin[first_task + segment] =
            begin;

        long_task_edge_end[first_task + segment] =
            end;
    }
}
```

---

## 5.5 可选优化：避免单lane串行生成大量segment

如果segment减小后，极长行可能生成很多task，当前单lane循环会成为UPDATE尾部。

可在第二阶段改成warp协作生成。

思路：

1. 计算warp中所有extreme row的segment count；
2. 对segment count做warp prefix；
3. lane 0一次性申请总task空间；
4. 全warp并行填写segment descriptor。

伪代码：

```cpp
local_segments =
    extreme_spike
        ? ceil_div(
            degree,
            LongSegmentSize)
        : 0;

segment_prefix_end =
    warp_prefix_sum(local_segments);

segment_prefix_begin =
    segment_prefix_end - local_segments;

warp_total_segments =
    __shfl_sync(
        FULL_MASK,
        segment_prefix_end,
        31);

if (lane == 0) {
    warp_task_base =
        atomicAdd(
            long_task_count,
            warp_total_segments);
}

warp_task_base =
    __shfl_sync(
        FULL_MASK,
        warp_task_base,
        0);

// warp cooperative fill
for (int logical_segment = lane;
     logical_segment < warp_total_segments;
     logical_segment += 32)
{
    owner_lane =
        warp_upper_bound(
            segment_prefix_end,
            logical_segment);

    owner_local_segment =
        logical_segment
        - prefix_begin_of(owner_lane);

    owner_row =
        warp_neuron_base
        + owner_lane;

    begin =
        row_begin(owner_lane)
        + owner_local_segment
        * LongSegmentSize;

    write_long_task(
        warp_task_base + logical_segment,
        owner_row,
        begin,
        min(begin + LongSegmentSize,
            row_end(owner_lane)));
}
```

第一轮segment size扫描先不做这项，以免调度方式和segment size同时变化，难以归因。

若128/256 segment导致UPDATE成本明显升高，再加入warp协作enqueue。

---

# 6. 三项优化之间的关系

## 6.1 普通 BlockTask edge-budget切分

主要解决：

```text
相似性/成本聚合
→ 单个普通BlockTask过重
→ BlockTask P99和尾部上升
```

它不减少active edges，也不减少atomic，只限制单任务上界。

## 6.2 Block reduce

主要解决：

```text
post相似性提高
→ 重复post增多
→ 但逐边atomic不变
```

它负责真正减少global atomic。

## 6.3 Long segment size

主要解决：

```text
极长行路径未被block化和重排覆盖
→ 旧的long-segment尾部仍存在
```

三者作用不同，必须分开做消融。

---

# 7. 推荐实施顺序

## 阶段 0：补充基线统计

在identity和最佳similarity方案上记录：

```text
普通BlockTask active-edge分布
long task edge分布
每时间步普通/long task数量
最后完成task类别
global atomic数量
每个task edges/unique posts
```

尤其需要知道阶段最后拖尾的是：

```text
普通重BlockTask
还是long segment
```

---

## 阶段 1：只加入普通task edge budget

先关闭reduce，保持long segment=1024。

扫描：

```text
BLOCK_EDGE_BUDGET =
128 / 256 / 512 / 1024 / unlimited
```

测试：

```text
identity
global similarity
当前最好的cost/similarity策略
```

验收：

* BlockTask P99明显下降；
* barrier/tail下降；
* task数增加不能过多；
* similarity不再显著慢于identity。

建议优先：

```text
256 / 512 / 1024
```

128可能使队列膨胀过快。

---

## 阶段 2：扫描long segment size

固定阶段1最佳block budget，reduce关闭。

扫描：

```text
128 / 256 / 512 / 1024 / 2048
```

记录：

* kernel时间；
* UPDATE时间；
* fanout时间；
* long task count；
* work-counter请求；
* barrier；
* Long Scoreboard。

选择端到端最佳值，而不是只看fanout阶段。

---

## 阶段 3：加入tile reduce

固定最佳：

```text
BLOCK_EDGE_BUDGET
LONG_SEGMENT_SIZE
```

比较：

```text
direct atomic
all-tile reduce
hinted tile reduce
```

分别测试：

```text
identity
global similarity
cost-constrained similarity
```

需要确认：

```text
相似性方案的global atomic下降幅度
>
identity方案
```

否则说明重排并未真正为reduce创造额外价值。

---

## 阶段 4：决定是否开发小hash

只有在以下条件成立时继续：

```text
tile reduce显示atomic减少能带来加速
且
高重复task中存在大量跨tile重复
```

否则保留tile reduce，不开发复杂hash。

---

# 8. 实验矩阵

为了避免组合爆炸，分三轮。

## 第一轮：Block budget

```text
reorder:
    identity
    global similarity

block budget:
    unlimited
    256
    512
    1024

long segment:
    1024

reduce:
    off
```

共8组。

---

## 第二轮：Long segment

```text
reorder:
    identity
    第一轮最佳similarity

block budget:
    第一轮最佳值

long segment:
    128
    256
    512
    1024
    2048

reduce:
    off
```

共10组。

---

## 第三轮：Reduce

```text
reorder:
    identity
    global similarity
    一个cost-constrained similarity

block budget:
    最佳值

long segment:
    最佳值

reduce:
    off
    all-tile
    hinted-tile
```

共9组。

---

# 9. 验收指标

## 普通task切分成功

应看到：

```text
BlockTask active-edge P99下降
max下降
barrier/tail下降
eligible warp提高
```

即使task数量增加，也应确保总时间下降。

## Long segment调优成功

应看到：

```text
long-tail等待下降
但queue和UPDATE开销没有过度上升
```

最优值不一定使long task最少，而是使：

[
T_{\text{enqueue}}
+
T_{\text{schedule}}
+
T_{\text{fanout}}
+
T_{\text{tail}}
]

最小。

## Reduce成功

至少满足：

```text
global atomic / active edge显著下降
Long Scoreboard下降
similarity + reduce优于identity + reduce
```

如果只有atomic数量下降，但kernel时间不降，则reduce本身的指令/同步开销仍然过高。

---

# 10. 预期风险

## 风险 1：普通task descriptor膨胀

block budget越小，队列越大。

应统计：

[
\text{tasks per original spike block}
]

若平均值远大于1，metadata成本可能抵消尾部收益。

## 风险 2：prefix二分重复执行

一个原始spike block被切成多个task后，每个task都要重新读取mask、degree并建立prefix。

如果切分过细，重复解析成本上升。

可在后续将prefix或row metadata缓存到更紧凑的descriptor，但第一版先不增加复杂度。

## 风险 3：tile reduce本身过重

`match_any_sync`可能比省下的少量atomic更贵。

因此必须保留：

```text
direct path
reduce path
```

并使用hint自适应选择。

## 风险 4：segment减小后UPDATE enqueue变慢

segment size从1024降到128时，极长行task数最多增加8倍。

如果最佳fanout segment较小但UPDATE显著变慢，应再实现warp协作enqueue。

---

# 11. 最小可行版本

第一版只实现：

```text
1. 普通BlockTask新增logical_begin/logical_end
2. UPDATE中根据total_active_edges切task
3. BLOCK_EDGE_BUDGET模板参数
4. LONG_SEGMENT_SIZE模板参数
5. 32-edge tile reduce开关
6. global atomic计数统计
```

暂不实现：

```text
流式逐laneflush
复杂shared hash
跨task reduce
动态运行时预算
跨warp/CTA reduce
warp specialization
TMA
```

---

# 12. 最终建议默认起点

建议从以下配置开始：

```text
BLOCK_EDGE_BUDGET = 512
LONG_SEGMENT_SIZE = 512
REDUCE = off
```

先验证任务尾部。

然后：

```text
REDUCE = hinted 32-edge tile reduce
```

在：

```text
identity
global similarity
```

上比较。

如果 similarity 的 global atomic 比identity明显下降，且kernel终于快于identity，说明完整链路成立：

```text
相似性重排
→ post重复增加
→ tile/block reduce
→ global atomic减少
→ Long Scoreboard下降
→ kernel加速
```

如果即使理想任务切分后，similarity + reduce仍不优于identity，则可以较有把握地判断：

> 当前 dominant-post 聚合产生的重复程度不足，继续优化该重排算法的优先级应降低。
