## 测试环境

测试数据集采用flybrain，在5090上测试，可进入服务器调试，是micromamba的ml-py312环境ssh -R 7897:127.0.0.1:7897 zhanghan@162.105.95.95，私钥如果缺失，可到本地win环境中寻找。运行在micromamba的ml-py312，可用GPU，调试可使用或参考benchmark/benchmark_rsnn_roofline.py

# Long-Segment Warp Specialization + TMA 第一版执行计划

## 1. 目标

针对 long-segment 任务，将当前串行路径：

```text
load segment edges
→ global atomic scatter
→ load next segment
```

改为：

```text
producer load chunk k+1
        并行
consumer atomic scatter chunk k
```

第一版只验证两阶段流水线：

```text
edge load
→ global atomic scatter
```

不实现：

* segment 内 hash；
* 独立 flush warp；
* CTA-wide aggregation；
* block task 流水线；
* 多级任务优先队列；
* 跨 timestep pipeline。

Long segment 天然满足：

```text
单条 CSR row
物理连续 edge span
100% edge 有效
固定或接近固定 segment size
无需 owner 查找
无需 map_packed_edge
```

因此它比 ordinary block task 更适合异步搬运。

---

# 2. 第一版需要回答的问题

第一版重点回答四个问题：

1. long-segment 路径在总 recurrent workload 中占比是否足够；
2. 少一个 consumer warp 的代价是否可接受；
3. 同步 shared staging 是否能控制在较小开销内；
4. edge load 是否能被后续 global atomic 延迟覆盖。

只有同步版本成立后，才进入 bulk async 和 TMA。

---

# 3. 保留不变的部分

保持现有：

* neuron update；
* spike emission；
* block task 构建；
* long-row 判定；
* long segment size = 512；
* ordinary block/hash 路径；
* timestep barrier；
* global atomic 写回语义；
* cooperative persistent launch；
* 当前数值容差。

Long-segment 优化只替换：

```text
process_long_segment_tasks()
```

ordinary block task 继续使用当前最佳 block/hash 版本。

---

# 4. 版本划分

建议使用编译期模式：

```cpp
BTORCH_LONG_WARP_SPEC_MODE
```

定义：

```text
0 = S0 原始 8-warp direct long-segment
1 = S1 7-warp direct，warp 0 空闲
2 = S2 producer 分发 descriptor，consumer direct load
3 = S3 同步 shared transport pipeline
4 = S4 bulk async pipeline
5 = S5 TMA pipeline
```

所有版本必须保持 ordinary block 路径相同。

---

# 5. 前置统计

在改动前先增加 long-segment 统计。

## 5.1 工作量统计

记录：

```text
long_task_count
long_edge_count

full_512_segment_count
partial_segment_count

segment_128_255
segment_256_383
segment_384_511
segment_512
```

派生：

```text
long edges / all recurrent edges
long tasks / all recurrent tasks
full-512 segment ratio
average edges / segment
```

## 5.2 时间占比统计

stats build 中使用 `clock64()` 粗略记录：

```text
long_path_cycles
block_path_cycles
update_path_cycles
```

只需用于估算，不要求精确 profiler 级别。

派生：

```text
long path cycle share
```

## 5.3 继续条件

建议：

```text
long edge coverage >= 15%
或
long path cycle share >= 10%
```

才进入完整 S3–S5。

若两者都很低，只实现 S1/S2 和简单实验即可。

---

# 6. 线程块角色划分

保持：

```text
256 threads
8 warps
```

S2 之后：

```text
warp 0:
    producer
    long-task scheduler
    chunk copy issuer

warp 1–7:
    consumer 0–6
    each processes one long segment
```

映射：

```cpp
constexpr int kProducerWarp = 0;
constexpr int kConsumerWarps = 7;

int warp_id = threadIdx.x >> 5;
int lane = threadIdx.x & 31;

bool is_producer = warp_id == 0;
bool is_consumer = warp_id > 0;

int consumer_id = warp_id - 1;
```

---

# 7. Segment task descriptor

每个 long segment 需要：

```cpp
struct LongSegmentTask {
    int batch;
    int edge_begin;
    int edge_count;
};
```

其中：

```text
0 < edge_count <= 512
```

来源：

```cpp
edge_begin =
    row_start + segment_id * 512;

edge_count =
    min(512, row_end - edge_begin);
```

每个 consumer 的 shared descriptor：

```cpp
__shared__ int consumer_batch[7];
__shared__ int consumer_edge_begin[7];
__shared__ int consumer_edge_count[7];

__shared__ int consumer_task_epoch[7];
__shared__ int consumer_done_epoch[7];
```

使用 block-scope release/acquire 发布。

---

# 8. 第一阶段 S1：7-warp direct

## 目标

单独测量：

```text
少一个 working warp 的成本
```

实现：

```cpp
if (warp_id == 0) {
    // long-segment phase idle
} else {
    while (true) {
        task = atomicAdd(long_task_counter, 1);

        if (task >= long_task_count)
            break;

        process_long_segment_direct(task);
    }
}
```

## 判定

相对 S0：

```text
regression <= 5%：继续
5%–8%：谨慎继续
>8%：专用 producer 不划算
```

若 S1 退化过大，后续考虑：

* producer 兼做 consumer；
* 每个 block 只部分 warp specialization；
* 无专用 producer 的 per-warp async pipeline。

---

# 9. 第二阶段 S2：descriptor specialization

## 目标

验证 producer 分发任务的协议成本。

Producer：

```text
给 idle consumer 分配一个 long segment descriptor
```

Consumer：

```text
收到 descriptor
→ 自己直接读 global post/weight
→ global atomic
→ done
```

不使用 shared edge buffer。

## Producer 伪代码

```cpp
producer_descriptor_loop():

    while tasks_remaining || any_consumer_active:

        progress = false

        for c in 0..6:

            if consumer_done_epoch[c]
               == consumer_task_epoch[c]:

                if tasks_remaining:

                    task = fetch_next_long_task()

                    batch =
                        long_task_batch(task)

                    edge_begin =
                        long_task_edge_begin(task)

                    edge_count =
                        long_task_edge_count(task)

                    consumer_batch[c] = batch
                    consumer_edge_begin[c] = edge_begin
                    consumer_edge_count[c] = edge_count

                    release_store(
                        consumer_task_epoch[c],
                        next_epoch(c)
                    )

                    progress = true

        if !progress:
            __nanosleep(64)
```

## Consumer 伪代码

```cpp
consumer_direct_loop(c):

    observed_epoch = 0

    while true:

        wait task_epoch[c] > observed_epoch
             or producer_done

        if producer_done
           && task_epoch[c] == observed_epoch:
            break

        acquire descriptor

        process_segment_direct(
            batch,
            edge_begin,
            edge_count
        )

        observed_epoch = task_epoch[c]

        release_store(
            consumer_done_epoch[c],
            observed_epoch
        )
```

## 判定

S2 相对 S1：

```text
额外 regression <= 3%
```

否则 producer descriptor 协议需要继续简化。

---

# 10. 第三阶段 S3：同步 transport pipeline

## 10.1 基本结构

采用：

```text
3 个 shared transport slot
每个 slot 128 edges
```

一个 512-edge segment拆为：

```text
4 × 128-edge chunks
```

每个 slot只负责：

```text
global → shared
consumer shared → register
```

consumer读完 slot后立即释放，再执行 global atomic。

---

## 10.2 Shared memory

```cpp
constexpr int kLongStages = 3;
constexpr int kChunkEdges = 128;

__shared__ int
    long_slot_post[kLongStages][kChunkEdges];

__shared__ float
    long_slot_weight[kLongStages][kChunkEdges];
```

metadata：

```cpp
__shared__ int long_slot_consumer[3];
__shared__ int long_slot_count[3];
__shared__ int long_slot_chunk_epoch[3];

__shared__ int long_slot_ready_epoch[3];
__shared__ int long_slot_consumed_epoch[3];
```

数据成本：

```text
3 × 128 × 8 B = 3 KB
```

加 metadata 后仍应很小。

目标：

```text
总 shared memory/block < 双 block residency 临界值
保持 2 blocks/SM
```

---

# 11. Consumer 状态

每个 consumer同一时间只拥有一个 segment。

维护：

```cpp
consumer_next_chunk[7];
consumer_chunk_count[7];
consumer_task_epoch[7];
consumer_done_epoch[7];
```

chunk count：

```cpp
chunk_count =
    ceil_div(edge_count, 128);
```

一般：

```text
edge_count 256–512
→ 2–4 chunks
```

小 segment可直接 fallback。

---

# 12. Pipeline eligibility

第一版建议：

```text
edge_count >= 256
→ pipeline

edge_count < 256
→ direct consumer path
```

之后 sweep：

```text
threshold = 128 / 256 / 384 / 512
```

记录：

```text
pipeline eligible tasks
pipeline eligible edges
pipeline edge coverage
```

与 block不同，long segment预计会有较高 coverage。

---

# 13. S3 Producer 逻辑

Producer负责：

1. 回收完成 consumer；
2. 给 idle consumer分配 segment；
3. 找一个需要chunk的 consumer；
4. 找一个 empty slot；
5. 整个 producer warp协作copy一个128-edge chunk；
6. 发布 slot ready；
7. 重复。

---

## 13.1 Producer 主循环伪代码

```cpp
producer_pipeline_loop():

    initialize_consumer_state()
    initialize_slots()

    while tasks_remaining || any_consumer_active:

        bool progress = false

        // A. 回收完成consumer
        for c in 0..6:

            if consumer_is_done(c):
                mark_consumer_idle(c)
                progress = true

        // B. 给idle consumer分配segment
        for c in round_robin_order:

            if !tasks_remaining:
                break

            if consumer_is_idle(c):

                task = fetch_next_long_task()

                publish_consumer_descriptor(
                    c,
                    task
                )

                consumer_next_chunk[c] = 0
                consumer_chunk_count[c] =
                    ceil_div(task.edge_count, 128)

                mark_consumer_active(c)

                progress = true

        // C. 生产一个chunk
        c = find_consumer_needing_chunk()
        slot = find_empty_slot()

        if c >= 0 && slot >= 0:

            produce_one_chunk(
                c,
                slot
            )

            consumer_next_chunk[c]++

            progress = true

        if !progress:
            producer_idle_counter++
            __nanosleep(64)

    release_store(producer_done, 1)
```

---

# 14. Consumer 选择策略

优先：

```text
已经拿到segment、正在等待下一个chunk的consumer
```

再分配新segment。

伪代码：

```cpp
find_consumer_needing_chunk():

    for offset in 0..6:

        c =
            (consumer_rr_cursor + offset) % 7

        if !consumer_active(c):
            continue

        if consumer_next_chunk[c]
           >= consumer_chunk_count[c]:
            continue

        if consumer_has_outstanding_chunk(c):
            continue

        consumer_rr_cursor =
            (c + 1) % 7

        return c

    return -1
```

一个 consumer同一时间最多有一个 outstanding chunk，简化协议。

---

# 15. Slot 选择

```cpp
find_empty_slot():

    for offset in 0..2:

        slot =
            (slot_rr_cursor + offset) % 3

        if slot_consumed_epoch[slot]
           == slot_ready_epoch[slot]:

            slot_rr_cursor =
                (slot + 1) % 3

            return slot

    return -1
```

不使用 consumer竞争 CAS。

slot由 producer唯一分配。

---

# 16. 同步 copy 伪代码

```cpp
produce_one_chunk(c, slot):

    chunk =
        consumer_next_chunk[c]

    source =
        consumer_edge_begin[c]
        + chunk * 128

    count =
        min(
            128,
            consumer_edge_count[c]
            - chunk * 128
        )

    for i = lane; i < count; i += 32:

        long_slot_post[slot][i] =
            graph_indices[source + i]

        long_slot_weight[slot][i] =
            graph_weight[source + i]

    __syncwarp()

    if lane == 0:

        long_slot_consumer[slot] = c
        long_slot_count[slot] = count

        chunk_epoch =
            make_chunk_epoch(
                consumer_task_epoch[c],
                chunk
            )

        long_slot_chunk_epoch[slot] =
            chunk_epoch

        release_store(
            long_slot_ready_epoch[slot],
            chunk_epoch
        )
```

Producer整个warp搬同一连续span，不允许lane-per-consumer串行copy。

---

# 17. Consumer 找到自己的 slot

每个 consumer等待：

```text
slot_consumer == consumer_id
并且
slot_chunk_epoch == expected_epoch
```

因为只有3个slot，扫描成本较小。

伪代码：

```cpp
wait_for_chunk_slot(c, expected_epoch):

    while true:

        for slot in 0..2:

            ready =
                acquire_load(
                    long_slot_ready_epoch[slot]
                )

            if ready != expected_epoch:
                continue

            if long_slot_consumer[slot] != c:
                continue

            return slot

        consumer_wait_counter++
        __nanosleep(64)
```

后续 TMA版本可以由 per-consumer slot id mailbox减少扫描，但第一版先保持简单。

---

# 18. Consumer 寄存器缓冲

每个128-edge chunk由一个warp读取。

每lane最多4条：

```cpp
int post_reg[4];
float weight_reg[4];
bool valid_reg[4];
```

伪代码：

```cpp
consume_chunk(c, slot):

    count =
        long_slot_count[slot]

    #pragma unroll
    for j in 0..3:

        index = lane + j * 32

        valid_reg[j] =
            index < count

        if valid_reg[j]:

            post_reg[j] =
                long_slot_post[slot][index]

            weight_reg[j] =
                long_slot_weight[slot][index]

    __syncwarp()

    // 此时shared数据全部进入register
    release_store(
        long_slot_consumed_epoch[slot],
        long_slot_ready_epoch[slot]
    )

    #pragma unroll
    for j in 0..3:

        if valid_reg[j]:

            atomicAdd(
                psc
                + batch * neuron_count
                + post_reg[j],
                weight_reg[j]
            )
```

关键顺序：

```text
shared → register
→ 释放slot
→ global atomic
```

这样 producer可以在 consumer做atomic时复用slot。

---

# 19. Consumer 主循环伪代码

```cpp
consumer_pipeline_loop(c):

    observed_task_epoch = 0

    while true:

        wait task_epoch[c] > observed_task_epoch
             or producer_done

        if producer_done
           && task_epoch[c] == observed_task_epoch:
            break

        acquire descriptor

        epoch =
            consumer_task_epoch[c]

        if edge_count < pipeline_threshold:

            process_segment_direct(
                batch,
                edge_begin,
                edge_count
            )

            release_store(
                consumer_done_epoch[c],
                epoch
            )

            observed_task_epoch = epoch
            continue

        chunk_count =
            ceil_div(edge_count, 128)

        for chunk in 0..chunk_count-1:

            expected =
                make_chunk_epoch(
                    epoch,
                    chunk
                )

            slot =
                wait_for_chunk_slot(
                    c,
                    expected
                )

            consume_chunk(
                c,
                slot
            )

        release_store(
            consumer_done_epoch[c],
            epoch
        )

        observed_task_epoch = epoch
```

---

# 20. Timestep drain

进入：

```cpp
grid.sync()
```

前必须满足：

```text
producer_done = true
所有 consumer done
所有 slots consumed
```

Producer发布done前：

```text
不再有未分配task
所有consumer已完成
所有slot已释放
```

伪代码：

```cpp
if producer:

    while tasks_remaining
       || any_consumer_active()
       || any_slot_outstanding():

        continue_pipeline()

    release_store(
        producer_done,
        1
    )

__syncthreads()
grid.sync()
```

必须增加最后一个chunk仍在atomic时的回归测试。

---

# 21. S4：bulk async版本

S3通过后，只替换：

```cpp
for i = lane; i < count; i += 32:
    global load
    shared store
```

为bulk async copy。

保持：

* descriptor协议；
* 3 slots；
* 128-edge chunk；
* consumer register-buffer；
* slot生命周期；
* fallback；
* task调度；

全部不变。

这样能隔离async copy本身的收益。

版本：

```text
S4-sync-state-machine + async-copy
```

---

# 22. S5：TMA版本

## 22.1 TMA输入

两个1D tensor map：

```text
post tensor map:
base = graph_indices
type = int32
length = total edges

weight tensor map:
base = graph_weight
type = float32
length = total edges
```

每次coordinate：

```text
edge_begin + chunk * 128
```

## 22.2 每slot barrier

```cpp
mbarrier long_slot_barrier[3];
```

每次slot复用维护phase：

```cpp
slot_phase[3];
```

Producer lane 0：

```cpp
if lane == 0:

    metadata[slot] = ...

    mbarrier_arrive_expect_tx(
        barrier[slot],
        bytes
    )

    issue_tma_post(
        post_map,
        source_coordinate,
        slot_post[slot],
        barrier[slot]
    )

    issue_tma_weight(
        weight_map,
        source_coordinate,
        slot_weight[slot],
        barrier[slot]
    )
```

Consumer：

```cpp
wait_mbarrier(
    barrier[slot],
    expected_phase
)
```

等待完成后读取shared。

partial chunk第一版建议：

```text
edge_count < 128
→ direct fallback
```

或只对完整128 chunk使用TMA，最后partial chunk同步copy。

---

# 23. 必须完成的消融实验

## 23.1 基础版本

| 版本 | 描述                                    |
| -- | ------------------------------------- |
| S0 | 8 warp direct                         |
| S1 | 7 warp direct                         |
| S2 | descriptor producer + direct consumer |
| S3 | sync 3-slot pipeline                  |
| S4 | bulk async                            |
| S5 | TMA                                   |

## 23.2 Chunk size sweep

测试：

```text
64
128
256
```

但每个配置必须记录shared memory和blocks/SM。

推荐优先顺序：

```text
128
64
256
```

256可能增加register数组：

```cpp
post_reg[8]
weight_reg[8]
```

导致寄存器压力明显上升。

## 23.3 Stage count

测试：

```text
2
3
4
```

预期3是较平衡点。

## 23.4 Pipeline threshold

测试：

```text
128
256
384
512
```

比较：

```text
eligible edge coverage
kernel time
producer wait
consumer wait
```

---

# 24. 数据集实验

至少测试：

```text
FlyBrain
Mice
```

以及：

```text
event rate:
0.005
0.020
0.100

batch:
1
4
```

重点报告：

```text
long edge coverage
full segment ratio
segment time share
speedup
```

long segment优化应在长尾fanout更强的数据集上收益更明显。

---

# 25. 软件统计

新增：

```text
LongTasks
LongEdges

LongPipelineEligibleTasks
LongPipelineEligibleEdges

LongFullSegments
LongPartialSegments

LongChunksProduced
LongChunksConsumed

LongProducerIdle
LongProducerNoSlot
LongProducerNoConsumer

LongConsumerTaskWait
LongConsumerChunkWait

LongMaxActiveConsumers
LongActiveConsumerCycleSum

LongSlotUse0
LongSlotUse1
LongSlotUse2
```

使用 `clock64()` 做时间加权active consumer：

```text
active_consumer_cycles
total_pipeline_cycles
```

避免简单producer loop采样偏差。

---

# 26. NCU对比

只采集：

```text
S0
S2
S3
S5
```

重点：

## 资源

```text
registers/thread
shared/block
blocks/SM
achieved occupancy
```

## 调度

```text
eligible warps/scheduler
issue active
active warps/scheduler
```

## Stall

```text
Long Scoreboard
Short Scoreboard
Barrier
MIO Throttle
LG Throttle
Wait
```

## 内存

```text
global load sectors
shared load/store
shared bank conflicts
global atomic/reduction throughput
L2 throughput
DRAM throughput
```

---

# 27. 正确性测试

## 27.1 Segment边界

构造：

```text
1 edge
127 edges
128 edges
129 edges
255 edges
256 edges
383 edges
384 edges
511 edges
512 edges
```

## 27.2 Task数量

```text
1 segment
2 segments
7 segments
8 segments
大量segments
```

## 27.3 并发

```text
consumer 0 atomic很慢
其他consumer持续执行
```

确认 producer不会固定等待 consumer 0。

## 27.4 Drain

测试：

```text
最后一个task
最后一个chunk
partial chunk
producer先完成分配
consumer仍在atomic
```

## 27.5 数值

比较：

```text
S0 vs S3
S0 vs S5
```

检查：

```text
spike mismatch
PSC max abs diff
PSC mean abs diff
PSC sum
PSC nonzero mismatch
v max abs diff
```

测试：

```text
1
8
32
128 timesteps
```

---

# 28. 性能门槛

## S1

```text
vs S0 regression <= 5%
```

## S2

```text
vs S1 regression <= 3%
```

## S3

必须满足：

```text
vs S2 regression <= 5%
pipeline edge coverage >= 20%
max active consumers >= 5
2 blocks/SM
```

更理想：

```text
S3接近或优于S2
```

## S4

```text
vs S3 speedup >= 3%
```

或：

```text
consumer chunk wait下降
Long Scoreboard下降
```

## S5

```text
vs S4 speedup >= 1–2%
```

最终目标：

```text
S5 vs S0 total kernel speedup >= 5%
```

---

# 29. 停止条件

出现以下情况则停止：

```text
1. long path cycle share <10%
2. pipeline edge coverage <20%
3. S1 regression >8%
4. S3 regression >5%
5. consumer长期等待producer
6. producer长期等待slot但consumer atomic完全主导
7. 3-slot增加导致1 block/SM
8. async只将Long Scoreboard转为Barrier/MIO
9. S4相对S3无收益
10. S5相对S4无增量
11. 最终总收益 <1–2%
```

---

# 30. 推荐实际编码顺序

## 第一天：统计与S1

完成：

```text
long edge/task coverage
full segment ratio
long path cycle share
S0/S1对照
```

## 第二步：S2 descriptor

完成：

```text
固定consumer ownership
producer descriptor分发
direct consumer
correctness + benchmark
```

## 第三步：S3同步pipeline

完成：

```text
3 × 128-edge slots
whole-warp cooperative copy
consumer register-buffer
early slot release
direct fallback
```

## 第四步：S3参数实验

测试：

```text
chunk 64/128/256
stages 2/3/4
threshold 128/256/384/512
```

确定最优同步结构。

## 第五步：S4 async

只替换copy。

## 第六步：S5 TMA

加入：

```text
tensor map
mbarrier
slot phase
TMA post/weight copy
```

最后采集完整NCU。

---

# 31. 第一版最终结构摘要

```text
Long-segment warp specialization

Warp 0:
    fetch and assign long segments
    manage 3 transport slots
    whole-warp sync copy in S3
    single-lane async/TMA issue in S4/S5

Warp 1–7:
    each owns one segment
    waits for 128-edge chunk
    loads full chunk into registers
    immediately releases shared slot
    performs global atomics from registers

Pipeline:
    producer loads chunk k+1
        overlaps
    consumer atomic-scatter chunk k

Shared memory:
    3 × 128 post/weight slots
    per-slot metadata and epoch/barrier
    per-consumer descriptor and epochs

Fallback:
    short partial segments direct
    ordinary block tasks unchanged

Critical invariants:
    source edge range physically continuous
    every loaded edge is valid
    slot is released before atomic phase
    producer never performs sparse gather mapping
    2 blocks/SM remains
```

该版本最关键的技术点是：

> consumer先把整个128-edge chunk读入寄存器，再释放shared slot，随后才执行global atomic。这样shared slot不会像旧block实现一样被昂贵的写回阶段长期占用，producer能够持续加载后续segment。
