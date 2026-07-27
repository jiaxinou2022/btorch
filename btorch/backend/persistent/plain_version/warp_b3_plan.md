# 修正版 B3：Block Continuous-Run Mailbox Pipeline 计划

## 1. 实验目标

修正版 B3 要回答的核心问题是：

> 对于输入 edge 已经物理连续、有效 edge 比例为 100% 的 block task，使用 producer warp 协作搬运和短生命周期 mailbox，能否在保留多个 consumer 并行度的情况下接近或超过 B2 direct path。

旧 B3 的 36.95% 回退不能直接视为 block pipeline 的结论，因为其实现同时存在：

1. producer lane 各自串行复制 64 条 edge；
2. producer warp 内不同 lane 执行不同 task 和不同控制流；
3. consumer 对已知全部有效的 edge 再做 owner 二分；
4. 每 64 edge 执行一次发布、等待和释放；
5. staged/direct 混合路径增加寄存器与控制流；
6. staged edge 覆盖率没有完整量化。

修正版只验证最干净的连续场景，不再试图覆盖一般离散 spike mask。

旧实验中的 B0/B1/B2 对照仍保留：

* B0：8 warp 原始 block/hash；
* B1：warp 0 空闲，7 warp direct；
* B2：warp 0 分发 descriptor，7 consumer direct；
* B3-clean：仅对满足条件的 task 启用修正版 mailbox，其余回退 B2。

B2 相对 B1 仅慢约 1.38%，说明 descriptor ownership 协议本身已基本可接受；B3-clean 应以 B2 为直接基线。

---

# 2. 第一版范围

## 2.1 只处理 eligible task

只有同时满足以下条件的 ordinary block task 才进入 B3-clean：

```text
1. active owner mask 是单个连续 run
2. task 使用显式 block-edge-budget descriptor
3. task edge 在 CSR 中是单个物理连续区间
4. active-edge ratio = 100%
5. task 总 edge 数 >= 256
6. task 总 edge 数 <= 512
```

其他情况全部走 B2 direct path：

```text
consumer 自行 map/load/hash/flush
```

第一版不处理：

* mask 内有 gap；
* first-active 到 last-active span 中包含 inactive row；
* 多段 CSR gather；
* full static 32-neuron block；
* owner metadata filtering；
* 小于 256 edge 的 task；
* long segment。

---

## 2.2 保留 hash 参数

保持当前实验最佳配置：

```text
aggregation = 512
capacity = 512
max probes = 4
min edges = 256
used slots = on
```

512-edge aggregation 的收益已被单独验证，不能为了简化 mailbox 改回 256-edge hash。

---

# 3. 修正版核心结构

线程块：

```text
256 threads
8 warps
```

角色：

```text
warp 0:
    producer / descriptor scheduler
    全 warp 协作搬运一个 mailbox chunk

warp 1–7:
    consumer 0–6
    各自拥有一个 mailbox
    各自拥有一个 512-entry hash
```

关键改变：

```text
旧 B3:
lane 0 串行复制 consumer 0 的 64 edges
lane 1 串行复制 consumer 1 的 64 edges
...

新 B3:
warp 0 的 32 lanes 协作复制一个 consumer 的 128-edge chunk
```

一个 128-edge chunk：

```cpp
for (int i = lane; i < count; i += 32) {
    mailbox_post[c][i] =
        graph_indices[source + i];

    mailbox_weight[c][i] =
        graph_weight[source + i];
}
```

每个 lane 最多处理 4 条 edge。

---

# 4. 为什么 mailbox 改为 128 edges

旧 B3 使用 64-edge mailbox。

一个 512-edge task需要：

```text
8 次 producer 发布
8 次 consumer 等待
8 次 consumer 释放
```

而每次只搬：

```text
64 × 8 B = 512 B
```

同步固定成本过高。

修正版使用：

```text
mailbox size = 128 edges
```

一个完整 512-edge task只需要：

```text
4 个 chunk
```

同步次数减半。

Shared-memory 成本：

```text
7 consumers
× 128 edges
× 8 B
= 7168 B
```

当前 B2 static shared memory约 39 KB；加入约 7 KB mailbox和少量 metadata 后，预计约 46–47 KB/block，应仍有机会保持 2 blocks/SM。B3 旧版为 42.7 KB/block，并已验证能够强制启动 340 cooperative blocks。

如果编译后超过双 block residency 临界值，则：

1. 先压缩 used-slot；
2. 再测试 96-edge mailbox；
3. 最后才回退 64；
4. 不降低 hash aggregation。

---

# 5. Mailbox 所有权

每个 consumer固定拥有自己的 mailbox：

```cpp
mailbox_post[7][128];
mailbox_weight[7][128];
mailbox_count[7];
```

同步状态：

```cpp
task_epoch[7];
mailbox_ready_epoch[7];
mailbox_consumed_epoch[7];
consumer_done_epoch[7];
```

不存在：

* consumer竞争共享stage；
* CAS抢占；
* 所有consumer扫描所有mailbox；
* producer固定等待某一个stage。

consumer只读取自己的状态。

---

# 6. Task descriptor

每个 consumer 的 descriptor：

```cpp
struct ConsumerTaskDescriptor {
    int batch;
    int edge_begin;
    int edge_count;
    int chunk_count;
    int task_epoch;
    int staged;
};
```

对于 B3-clean task：

```text
edge_begin = CSR contiguous span begin
edge_count = active edge count
chunk_count = ceil(edge_count / 128)
staged = true
```

对于 fallback task：

```text
保留 B2 原 descriptor
staged = false
```

descriptor 发布使用 block-scope release/acquire。

---

# 7. Producer 调度原则

producer warp一次只协作搬运一个 chunk，但可在七个 consumer之间轮转。

producer维护：

```cpp
next_chunk[7];
active_task[7];
```

每轮扫描所有 consumer：

1. 给 IDLE consumer分配新 task；
2. 给 mailbox 已消费的 staged consumer搬下一个chunk；
3. 回收已完成 consumer；
4. 若无进展则短暂 backoff。

不能固定：

```text
consumer 0 未完成
→ producer 一直等待 consumer 0
```

而应：

```text
consumer 0 mailbox busy
→ 检查 consumer 1–6
```

---

# 8. Producer 伪代码

```cpp
function producer_loop():

    initialize_consumer_state()

    while tasks_remaining() || any_consumer_active():

        bool progress = false

        // A. 回收完成任务
        for c in round_robin(0 .. 6):

            if consumer_done_epoch[c]
                    == assigned_task_epoch[c]:

                mark_consumer_idle(c)
                progress = true

        // B. 向空闲 consumer 分配 task
        for c in round_robin(0 .. 6):

            if !tasks_remaining():
                break

            if consumer_is_idle(c):

                task = fetch_next_task()

                if is_b3_clean_eligible(task):
                    publish_staged_descriptor(c, task)
                    next_chunk[c] = 0
                else:
                    publish_direct_descriptor(c, task)

                mark_consumer_active(c)
                progress = true

        // C. 整个 producer warp 协作填一个 mailbox
        int selected = find_any_consumer_needing_chunk()

        if selected >= 0:

            copy_one_chunk_cooperatively(selected)

            next_chunk[selected]++

            progress = true

        if !progress:
            producer_backoff()

    publish_producer_done()
```

---

## 8.1 选择 consumer

```cpp
function find_any_consumer_needing_chunk():

    for offset in 0 .. 6:

        c = (round_robin_cursor + offset) % 7

        if !consumer_has_staged_task(c):
            continue

        if next_chunk[c] >= chunk_count[c]:
            continue

        if mailbox_consumed_epoch[c]
                != expected_previous_epoch(c):
            continue

        round_robin_cursor = (c + 1) % 7
        return c

    return -1
```

只由 lane 0 决定 `selected`，然后广播：

```cpp
selected = __shfl_sync(FULL_MASK, selected, 0);
```

---

## 8.2 协作搬运

```cpp
function copy_one_chunk_cooperatively(c):

    int chunk = next_chunk[c]

    int source =
        task_edge_begin[c]
        + chunk * 128

    int count =
        min(128,
            task_edge_count[c] - chunk * 128)

    for i = lane; i < count; i += 32:

        mailbox_post[c][i] =
            graph_indices[source + i]

        mailbox_weight[c][i] =
            graph_weight[source + i]

    __syncwarp()

    if lane == 0:

        mailbox_count[c] = count

        publish_release(
            mailbox_ready_epoch[c],
            make_chunk_epoch(
                assigned_task_epoch[c],
                chunk
            )
        )
```

producer warp所有 lane执行同一 consumer、同一 source span，不再发生 lane-per-consumer divergence。

---

# 9. Consumer 伪代码

```cpp
function consumer_loop(c):

    int observed_task_epoch = 0

    while true:

        wait_until(
            task_epoch[c] > observed_task_epoch
            || producer_done
        )

        if producer_done
           && task_epoch[c] == observed_task_epoch:
            break

        descriptor =
            acquire_task_descriptor(c)

        observed_task_epoch =
            descriptor.task_epoch

        if !descriptor.staged:

            process_direct_b2_task(descriptor)

            publish_consumer_done(c)
            continue

        hash_prepare(c)

        for chunk in 0 .. descriptor.chunk_count - 1:

            expected =
                make_chunk_epoch(
                    descriptor.task_epoch,
                    chunk
                )

            wait_until(
                mailbox_ready_epoch[c]
                == expected
            )

            int count =
                acquire_mailbox_count(c)

            process_mailbox_chunk(
                c,
                count,
                descriptor.batch
            )

            publish_release(
                mailbox_consumed_epoch[c],
                expected
            )

        flush_hash(
            c,
            descriptor.batch
        )

        publish_consumer_done(c)
```

---

# 10. 删除 owner 二分

旧 B3 对每个 edge执行五轮 prefix二分，以寻找 owner neuron。

修正版 B3-clean 已经要求：

```text
active mask 是连续run
CSR span只包含active owner
active-edge ratio = 100%
```

所以 mailbox中的每条 valid edge都必然有效。

consumer直接：

```cpp
bool valid = index < count;

int post =
    valid
        ? mailbox_post[c][index]
        : 0;

float weight =
    valid
        ? mailbox_weight[c][index]
        : 0.0f;
```

不再计算：

```text
relative edge
packed prefix
binary search owner
spike mask lookup
```

同时 staged consumer不再构造 packed prefix。

这一点必须通过 debug assertion验证，而不能在性能版本里继续重复计算。

---

## 10.1 Debug 验证

在 debug build中可抽样验证：

```cpp
assert(task_edge_count ==
       sum(degree of active contiguous run));
```

以及：

```cpp
assert(task_edge_begin ==
       graph_indptr[block_start + first_active]);
```

```cpp
assert(task_edge_begin + task_edge_count ==
       graph_indptr[block_start + last_active + 1]);
```

验证通过后，release build删除 owner定位。

---

# 11. Hash chunk 处理

consumer保留跨 chunk 的 hash状态。

```cpp
function process_mailbox_chunk(
    consumer,
    count,
    batch
):

    for base = 0; base < count; base += 32:

        int index = base + lane
        bool valid = index < count

        if valid:
            post =
                mailbox_post[consumer][index]

            weight =
                mailbox_weight[consumer][index]

        hash_insert_or_fallback(
            consumer,
            valid,
            post,
            weight,
            batch
        )
```

处理完一个128-edge chunk后：

```text
只释放 mailbox
不 flush hash
不清理 hash
```

完整 task的所有chunk处理完后：

```text
flush hash
清理 used slots
```

由此仍保持512-edge聚合范围。

---

# 12. 同步实现

第一版不使用 `__threadfence_block()` 模拟 acquire。

推荐：

```cpp
cuda::atomic_ref<
    int,
    cuda::thread_scope_block
>
```

producer发布 mailbox：

```cpp
ready_ref.store(
    epoch,
    cuda::memory_order_release
);
```

consumer等待：

```cpp
while (
    ready_ref.load(
        cuda::memory_order_acquire
    ) != epoch
) {
    __nanosleep(backoff);
}
```

consumer释放：

```cpp
consumed_ref.store(
    epoch,
    cuda::memory_order_release
);
```

producer读取：

```cpp
consumed_ref.load(
    cuda::memory_order_acquire
);
```

不使用：

```text
atomicAdd(x, 0)
atomicCAS抢占
全局stage扫描
```

---

# 13. 编译期版本

建议新增：

```cpp
BTORCH_WARP_SPEC_MODE
```

模式：

```text
0 = B0 original
1 = B1 seven-warp direct
2 = B2 descriptor mailbox
3 = old B3, optional retained for comparison
4 = B3-clean sync 128
```

后续：

```text
5 = B3-clean async 128
6 = B3-clean TMA 128
```

不要直接覆盖旧 B3，保留用于验证修复收益来源。

---

# 14. 新增统计

## 14.1 Coverage

必须记录：

```text
ordinary_tasks
ordinary_edges

b3_eligible_tasks
b3_eligible_edges

b3_staged_tasks
b3_staged_edges

b3_fallback_tasks
b3_fallback_edges
```

派生：

```text
eligible task coverage
eligible edge coverage
staged edge coverage
```

最重要的是：

[
\text{staged edge coverage}
===========================

\frac{\text{B3 staged edges}}
{\text{ordinary block edges}}
]

---

## 14.2 Task size

记录：

```text
tasks with:
256–383 edges
384–511 edges
512 edges
```

记录：

```text
1-chunk tasks
2-chunk tasks
3-chunk tasks
4-chunk tasks
```

因为同步摊销高度依赖 task大小。

---

## 14.3 Pipeline

记录：

```text
producer task-dispatch waits
producer mailbox waits
producer idle loops

consumer task waits
consumer mailbox waits

mailbox chunks produced
mailbox chunks consumed

max simultaneous active consumers
average active consumers
```

关键目标：

```text
max active consumers >= 5
average active consumers尽量接近5–7
```

---

## 14.4 时间拆分

建议用 `clock64()` 采样累计：

```text
producer descriptor cycles
producer copy cycles
producer wait cycles

consumer mailbox wait cycles
consumer hash cycles
consumer flush cycles
```

只在 stats build启用。

不要在每次循环内global atomic计数；使用 warp-local或block-local累计，timestep结束后统一写回。

---

# 15. 正确性实验

## 15.1 单元测试

构造以下 contiguous-run task：

```text
1. 256 edges
2. 257 edges
3. 383 edges
4. 384 edges
5. 511 edges
6. 512 edges
7. 最后一个partial 128-edge chunk
8. 大量重复post
9. 完全无重复post
10. probe fallback
```

---

## 15.2 Eligibility测试

验证：

```text
contiguous mask → staged
mask with one gap → fallback
mask with multiple runs → fallback
edge_count < 256 → fallback
invalid explicit descriptor → fallback
```

---

## 15.3 并发测试

构造：

```text
consumer 0 flush极慢
consumer 1–6持续收到任务
```

验证 producer不会固定等待consumer 0。

构造七个同时eligible任务，确认：

```text
max simultaneous active consumers >= 5
```

---

## 15.4 数值比较

优先比较：

```text
B3-clean vs B2
```

逐 timestep检查：

```text
spike mismatch
PSC max abs diff
PSC mean abs diff
PSC sum diff
PSC nonzero-position mismatch
v max abs diff
```

测试：

```text
1 timestep
8 timesteps
32 timesteps
128 timesteps
```

同时验证：

```text
B2 ordinary input edges
=
B3 staged active edges
+ B3 fallback processed edges
```

---

# 16. 性能实验设计

## 16.1 固定环境

保持此前配置：

```text
GPU: RTX 5090
dataset: FlyBrain
batch: 1
timesteps: 128
warmup: 10
repeat: 50

block_edge_budget: 256
long_segment_size: 512

hash aggregation: 512
hash capacity: 512
max probes: 4
min edges: 256

reorder: global_similarity
event rate: 0.02
```

服务器繁忙时，采用 interleaved benchmark：

```text
B2
B3-clean
B2
B3-clean
```

每轮均记录50次中位数。

---

## 16.2 第一组：工程修复消融

依次比较：

| 版本       | 改动                                          |
| -------- | ------------------------------------------- |
| B2       | descriptor only，direct edge load            |
| B3-old   | lane-per-consumer，64-edge mailbox           |
| B3-C1    | producer warp协作copy，64-edge，保留owner lookup  |
| B3-C2    | producer warp协作copy，64-edge，删除owner lookup  |
| B3-clean | producer warp协作copy，128-edge，删除owner lookup |

这组实验回答：

1. producer协作copy贡献多少；
2. owner二分贡献多少；
3. mailbox从64到128贡献多少。

如果时间有限，至少测试：

```text
B2
B3-old
B3-C2
B3-clean
```

---

## 16.3 第二组：Coverage分层

按 staged edge coverage分组统计或构造：

```text
低 coverage
中 coverage
高 coverage
```

也可对event rate进行 sweep：

```text
0.005
0.020
0.100
```

观察 contiguous-run task覆盖率是否随活动率变化。

---

## 16.4 第三组：Mailbox size

测试：

```text
64
96
128
```

不默认测试256，因为 shared memory可能破坏2 blocks/SM。

每个配置同时记录：

```text
shared memory/block
registers/thread
active blocks/SM
```

---

## 16.5 第四组：Reorder

测试：

```text
identity
global_similarity
```

global similarity可能提高post重复度，但也可能改变active run和block长度分布。

必须分别报告：

```text
staged edge coverage
hash atomic reduction
kernel time
```

---

## 16.6 第五组：活动率与batch

至少测试：

| event rate | batch |
| ---------: | ----: |
|      0.005 |     1 |
|      0.020 |     1 |
|      0.100 |     1 |
|      0.020 |     4 |

此前512-edge hash在这些配置下均有正收益，因此 B3-clean不应只在单一活动率下测试。

---

# 17. NCU实验

只对以下三个版本采集完整 NCU：

```text
B2
B3-old
B3-clean
```

## 17.1 资源

```text
registers/thread
static shared/block
active blocks/SM
achieved occupancy
```

硬性要求：

```text
2 blocks/SM
```

---

## 17.2 Warp调度

```text
eligible warps/scheduler
active warps/scheduler
issue active
warp execution efficiency
branch efficiency
```

预期：

```text
B3-clean producer warp execution efficiency
显著高于B3-old
```

---

## 17.3 Stall

```text
Long Scoreboard
Short Scoreboard
Barrier
MIO Throttle
LG Throttle
Wait
```

理想变化：

```text
producer divergence下降
Long Scoreboard下降或不升
Barrier不显著增加
MIO Throttle不显著增加
```

---

## 17.4 内存路径

```text
global load sectors
shared load/store instructions
shared bank conflicts
DRAM bytes
L2 bytes
```

B3-clean不应像早期full-span版本一样显著增加DRAM读取量。

验证：

```text
loaded staged bytes
≈ staged active edge bytes
```

---

# 18. 成功和停止标准

## 18.1 正确性门槛

必须满足：

```text
spike mismatch = 0
无 task遗漏
无 task重复
无 mailbox epoch错误
无 timestep drain错误
```

PSC误差不得显著高于 B2正常浮点原子顺序差异。

---

## 18.2 架构门槛

必须满足：

```text
2 blocks/SM
max simultaneous active consumers >= 5
无 local stack
producer协作copy无严重divergence
```

---

## 18.3 性能门槛

相对 B2：

```text
B3-clean regression <= 5%
```

才允许进入 async copy阶段。

更理想：

```text
B3-clean <= B2
```

如果同步 staging已接近打平，则 async copy可能提供增量收益。

---

## 18.4 停止条件

以下任一成立，则停止 block pipeline：

```text
1. B3-clean仍比B2慢 >5%
2. producer协作copy后consumer仍大量等待
3. staged edge coverage <20%
4. shared memory导致1 block/SM
5. Barrier/MIO stall抵消Long Scoreboard下降
6. mailbox 64/96/128均无改善
7. 高coverage场景仍明显负优化
```

其中最有决定性的是：

```text
高 staged-edge coverage
+ cooperative copy
+ no owner search
+ 128-edge mailbox
```

在此条件下仍负优化。

---

# 19. 进入异步copy的条件

只有 B3-clean通过同步门槛后，才实现：

```text
B4 = cp.async/bulk async
```

替换部分仅限：

```text
producer cooperative global→shared copy
```

其他调度、hash、fallback全部保持不变。

比较：

```text
B3-clean sync
vs
B4 async
```

若 B4相对B3-clean：

```text
加速 >=3%
```

或 Long Scoreboard明显下降，则继续 TMA。

---

# 20. 进入TMA的条件

只有满足：

```text
source连续
task通常>=256 edges
staged coverage足够
B4已有正收益
```

才实现：

```text
B5 = TMA
```

TMA tile先使用：

```text
128 edges
post和weight分别搬运
```

若128过小，再考虑把两个chunk合并为256-edge TMA transaction，但 shared mailbox仍可分批消费。

---

# 21. 最终实验表格

最终报告至少包含：

| Version      | us/step |    vs B2 | Reg/thread | Shared/block | Active blocks/SM |
| ------------ | ------: | -------: | ---------: | -----------: | ---------------: |
| B0           |         |          |            |              |                  |
| B1           |         |          |            |              |                  |
| B2           |         | baseline |            |              |                  |
| B3-old       |         |          |            |              |                  |
| B3-C2        |         |          |            |              |                  |
| B3-clean-64  |         |          |            |              |                  |
| B3-clean-96  |         |          |            |              |                  |
| B3-clean-128 |         |          |            |              |                  |

Coverage表：

| event rate | eligible task % | eligible edge % | avg task edges | full-512 % |
| ---------: | --------------: | --------------: | -------------: | ---------: |

Pipeline表：

| Version | producer wait | consumer wait | avg active consumers | max active consumers |
| ------- | ------------: | ------------: | -------------------: | -------------------: |

NCU表：

| Metric                    | B2 | B3-old | B3-clean |
| ------------------------- | -: | -----: | -------: |
| Eligible warps/scheduler  |    |        |          |
| Issue active              |    |        |          |
| Long Scoreboard           |    |        |          |
| Barrier                   |    |        |          |
| MIO Throttle              |    |        |          |
| Warp execution efficiency |    |        |          |
| Global load bytes         |    |        |          |

---

# 22. 最终实现摘要

```text
B3-clean

Eligible task:
    contiguous active-owner run
    100% active edge ratio
    256–512 edges
    physically contiguous CSR span

Producer warp:
    lane 0调度
    全32 lanes协作copy
    每次搬128 edges
    扫描任意可服务consumer
    不执行owner mapping
    不按lane绑定consumer

Consumer warp:
    固定私有mailbox
    固定私有512-entry hash
    每个chunk处理后立即释放mailbox
    完整task结束后flush
    staged路径不做owner二分

Fallback:
    不符合条件的task走B2 direct path

Primary decision:
    B3-clean相对B2是否能控制在5%以内
```

这次实验的最终价值不只是争取加速，而是建立一个可信的停止结论：

> 如果在连续输入、100%有效edge、producer warp协作copy、无owner查找、较粗同步粒度和2 blocks/SM都成立时，B3-clean仍显著慢于B2，那么可以有把握地认为当前 block task 的工作粒度不足以支撑额外的 shared staging pipeline，并将后续流水线工作转向 long segment。
