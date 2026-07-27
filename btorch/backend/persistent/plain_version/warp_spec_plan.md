## 调试环境

测试数据集采用flybrain，在5090上测试，可进入服务器调试，是micromamba的ml-py312环境ssh -R 7897:127.0.0.1:7897 zhanghan@162.105.95.95，私钥如果缺失，可到本地win环境中寻找。运行在micromamba的ml-py312，可用GPU，调试可使用或参考benchmark/benchmark_rsnn_roofline.py

# Warp Specialization 新第一版实施计划

## 1. 第一版定位

第一版目标不是直接完成：

```text
TMA edge load
→ independent hash accumulate
→ independent global flush
```

而是先验证一个更基础的问题：

> 在不降低当前 512-edge hash 路径任务并行度的前提下，能否将连续 edge tile 的加载与 hash/flush 重叠。

新版本必须满足：

1. 保持多个 consumer warp 同时处理不同 512-edge task；
2. producer 不执行逐 edge 的 `map_packed_edge()` 和离散 gather；
3. shared stage 只保存正在搬运的数据，不覆盖整个 hash/flush 生命周期；
4. consumer 获取数据后立即释放 stage；
5. 不使用多个 consumer 竞争两个 stage 的结构；
6. 不用高频 shared atomic 轮询；
7. 保持 2 blocks/SM；
8. 第一版只做 load 与 compute 两阶段，不拆独立 flush。

---

# 2. 上一版失败带来的设计约束

## 2.1 Stage 数量不能限制 active consumer 数量

上一版中，一个 stage 从：

```text
READY
→ BUSY
→ hash
→ global flush
→ EMPTY
```

在整个任务期间被 consumer 占用。

两个 stage 因而最多允许两个 consumer 工作，即使线程块中有七个 consumer warp。

新版本必须改为：

```text
stage READY
→ consumer 将数据读取到寄存器或私有执行状态
→ stage 立即释放
→ consumer 继续 hash/flush
```

因此，stage 的生命周期只覆盖：

```text
producer 写入
→ consumer 读取
```

而不覆盖：

```text
hash
→ global flush
```

---

## 2.2 Producer 不能成为 gather server

上一版 producer 对所有任务执行：

```text
prefix construction
→ map_packed_edge × 512
→ graph_indices gather × 512
→ graph_weight gather × 512
→ shared store × 512
```

这把原本由多个 warp 并行执行的工作集中到了一个 warp。

新版本中，producer必须只处理：

```text
读取 task descriptor
→ 计算连续 source offset
→ 发起异步 bulk copy
→ 发布完成状态
```

因此，新第一版的前置条件是：

> 输入 edge tile 在物理上必须连续。

如果 edge仍然需要 `map_packed_edge()` 才能访问，则暂时不引入专用 producer。

---

## 2.3 不使用竞争式共享队列

上一版七个 consumer不断：

```text
CAS(stage 0)
CAS(stage 1)
read producer_done
read stage state
sleep
retry
```

产生大量无效 shared atomic。

新版本采用静态 ownership：

```text
每个 consumer warp拥有自己的 mailbox
```

consumer不与其他 warp竞争。

---

# 3. 新第一版总体结构

线程块仍为：

```text
256 threads
8 warps
```

角色划分：

```text
warp 0:
    producer / scheduler

warp 1–7:
    consumer 0–6
```

每个 consumer拥有：

1. 一份 warp-private 512-entry hash；
2. 一个固定的 shared-memory mailbox；
3. 一个固定的 producer-consumer barrier；
4. 一个正在处理的 512-edge task；
5. 自己的 task epoch。

总体结构：

```text
                    ┌── mailbox 0 → consumer 0 → hash/flush
producer/scheduler ─┼── mailbox 1 → consumer 1 → hash/flush
                    ├── mailbox 2 → consumer 2 → hash/flush
                    ├── ...
                    └── mailbox 6 → consumer 6 → hash/flush
```

不存在：

```text
7 consumers compete for 2 stages
```

而是：

```text
1 producer feeds 7 fixed consumers
```

---

# 4. 数据布局前置改造

## 4.1 目标布局

为 ordinary block task 建立物理连续的 edge tile：

```cpp
packed_task_offsets[num_tasks + 1];
packed_post[total_packed_edges];
packed_weight[total_packed_edges];
```

每个 task descriptor包含：

```cpp
struct PackedTask {
    int batch;
    int packed_offset;
    int edge_count;
};
```

其中：

```text
0 < edge_count <= 512
```

producer不再需要：

```cpp
map_packed_edge()
```

而是直接得到：

```cpp
source_post   = packed_post   + packed_offset;
source_weight = packed_weight + packed_offset;
```

---

## 4.2 预处理原则

当前性能最好的 hash粒度仍保留为 512 edges。先前实验已经表明，512-edge task合并本身带来约 4.6% 收益，hash相对匹配的 512-edge direct版本还可继续加速约 10.4%；不能为了流水线退回 256-edge聚合。

预处理将每个活跃 block可能访问的 CSR row连续打包。

第一版可以容忍一定的数据复制，但必须记录：

```text
packed edge storage / original CSR edge storage
```

避免预处理形成不可接受的存储膨胀。

---

## 4.3 第一版不处理动态 spike mask压缩

如果运行时 spike mask决定哪些 row有效，第一版不尝试为所有 mask预生成 packed tile。

更合适的第一版范围是：

### 路径 A：只处理已经物理连续的任务

例如：

* long segment；
* 单一长 row；
* 重排后连续范围；
* 已在 task构建阶段形成连续 packed task。

### 路径 B：离线生成 block packed layout，并携带 owner lane

布局：

```cpp
packed_post[];
packed_weight[];
packed_owner_lane[];
```

consumer读取 edge后判断：

```cpp
active = spike_mask & (1u << owner_lane);
```

该方案会搬运未发放 neuron的 edge，因此需要统计：

```text
active edge ratio =
active edges / loaded packed edges
```

第一版只有在该比例达到合理水平时才启用。

建议停止线：

```text
median active edge ratio < 50%
```

则不对该类 task使用 TMA，而回退现有直接路径。

---

# 5. Mailbox 设计

## 5.1 Mailbox 不保存完整 512-edge task

每个 consumer的 mailbox只保存一个较小 chunk：

```text
64 edges
```

七个 mailbox的 shared-memory成本：

```text
7 × 64 × (4 B post + 4 B weight)
= 3584 B
```

如果还需要 owner：

```text
7 × 64 × 1 B
≈ 448 B
```

总增量约 4 KB。

这比两个完整 512-edge stage的 8 KB更小，而且不会把 active consumer限制为两个。

---

## 5.2 Consumer 在一个 hash窗口中接收多个 chunk

一个 512-edge task拆为：

```text
8 × 64-edge chunks
```

consumer收到第一个 chunk后初始化本轮 hash：

```text
chunk 0
→ hash accumulate
→ release mailbox

chunk 1
→ hash accumulate
→ release mailbox

...

chunk 7
→ hash accumulate
→ release mailbox
→ global flush
```

关键点：

> Mailbox处理完 64 edges后立即释放，但 consumer自己的 hash状态保留到完整512-edge task结束。

因此：

```text
mailbox lifetime ≈ one 64-edge chunk
```

而不是：

```text
mailbox lifetime ≈ 512-edge hash + global flush
```

---

# 6. Mailbox 状态

每个 consumer拥有独立状态：

```cpp
enum MailboxState {
    MAILBOX_EMPTY,
    MAILBOX_LOADING,
    MAILBOX_READY
};
```

建议额外使用 epoch避免 ABA问题：

```cpp
mailbox_ready_epoch[7];
mailbox_consumed_epoch[7];
```

producer写第 `e` 个 chunk后：

```cpp
ready_epoch[c] = e;
```

consumer处理完成后：

```cpp
consumed_epoch[c] = e;
```

producer只检查对应 consumer的 consumed epoch，不扫描其他 consumer的 mailbox状态。

---

## 6.1 不使用全局竞争 CAS

producer已经明确知道任务分给哪个 consumer，因此不需要：

```cpp
atomicCAS(any stage)
```

consumer也只等待自己的 mailbox：

```cpp
wait ready_epoch[consumer_id] == expected_epoch
```

同步应使用：

```cpp
cuda::atomic_ref<int, cuda::thread_scope_block>
```

或后续 TMA使用的 `mbarrier`。

producer发布：

```cpp
ready.store(epoch, memory_order_release);
```

consumer读取：

```cpp
while (ready.load(memory_order_acquire) != epoch) {
    backoff();
}
```

consumer释放：

```cpp
consumed.store(epoch, memory_order_release);
```

---

# 7. Producer 调度策略

producer需要让七个 consumer持续有任务，但不负责实际 hash。

每个 consumer维护：

```cpp
consumer_task[c];
consumer_chunk[c];
consumer_epoch[c];
consumer_busy[c];
```

producer使用 round-robin寻找可用 consumer：

```text
consumer 0
consumer 1
...
consumer 6
consumer 0
...
```

但不能像上一版一样固定等待某一个 consumer。

应扫描任意可用 consumer：

```cpp
for c in round-robin order:
    if mailbox free and consumer can accept chunk:
        dispatch
```

如果当前 consumer正在 global flush，其 mailbox可能仍然空，但不能给它下一个 task，除非允许双 task状态。

第一版保持简单：

```text
一个 consumer同一时间只拥有一个 512-edge task
```

producer等待该 consumer完成整项任务后，才给它分配新 task。

不过在同一任务内部，可以持续发送下一个64-edge chunk。

---

# 8. Consumer 状态机

每个 consumer独立执行：

```text
IDLE
→ TASK_ASSIGNED
→ RECEIVE_CHUNKS
→ HASH_ACCUMULATE
→ FLUSH
→ IDLE
```

consumer状态：

```cpp
enum ConsumerState {
    CONSUMER_IDLE,
    CONSUMER_PROCESSING,
    CONSUMER_DONE
};
```

producer只有在：

```text
consumer_state == IDLE
```

时分配新 task。

任务分配包含：

```cpp
task_offset[c];
task_edge_count[c];
task_batch[c];
task_spike_mask[c];
task_epoch[c];
```

producer写 descriptor并 release发布。

---

# 9. 第一阶段实现：不使用 TMA

为了先验证新的调度结构，第一阶段仍使用普通向量化连续 copy，但必须保证 source连续。

producer warp协同搬运64-edge chunk：

```cpp
for i = lane; i < chunk_edges; i += 32:
    mailbox_post[c][i] =
        packed_post[offset + chunk_begin + i];

    mailbox_weight[c][i] =
        packed_weight[offset + chunk_begin + i];
```

注意，这与上一版有本质区别：

上一版：

```text
producer执行512次map + gather
```

新版本：

```text
producer执行64条连续load/store
```

并且 producer在 consumer处理该 chunk期间可以为其他 consumer填 mailbox。

这一阶段命名：

```text
WS-mailbox-sync
```

它验证：

* 七 consumer是否可以同时运行；
* mailbox生命周期是否足够短；
* producer是否能持续供给；
* 4 KB shared增量是否保持2 blocks/SM；
* 固定 ownership是否消除了 polling storm。

---

# 10. 第二阶段实现：异步 copy

同步版本正确且没有明显退化后，将64-edge连续 copy替换为：

```text
cp.async bulk
```

或对应架构支持的异步 bulk copy。

版本：

```text
WS-mailbox-async
```

producer流程：

```text
wait mailbox empty
→ issue async copy
→ commit
→ copy complete
→ publish mailbox ready
```

第一版不要求 producer一次发起多个未完成 transaction；先确保单 mailbox异步 copy正确。

---

# 11. 第三阶段实现：TMA

只有当 source确实物理连续且 bulk async版本有效，才替换为TMA。

TMA第一版只处理一维连续数组：

```text
post tile
weight tile
```

不设计复杂二维 tensor map。

每个 consumer mailbox对应独立 barrier：

```cpp
mbarrier mailbox_barrier[7];
```

producer：

```text
arrive_expect_tx
→ issue TMA post copy
→ issue TMA weight copy
```

consumer：

```text
wait barrier completion
→ consume mailbox
```

版本：

```text
WS-mailbox-TMA
```

---

# 12. 核心伪代码

## 12.1 Kernel主流程

```cpp
for t in timesteps:

    apply_external_input()
    grid.sync()

    update_neurons_and_emit_spikes()
    build_tasks()
    grid.sync()

    process_long_tasks_original_path()

    __syncthreads()

    initialize_warp_specialization_state()

    if warp_id == 0:
        producer_loop()
    else:
        consumer_loop(consumer_id)

    __syncthreads()

    verify_all_consumers_idle()
    verify_all_mailboxes_consumed()

    grid.sync()
```

---

## 12.2 Producer主循环

```cpp
function producer_loop():

    next_task = 0
    remaining_tasks = ordinary_task_count

    while remaining_tasks > 0 or any_consumer_busy():

        // 1. 给空闲consumer分配新任务
        for c in round_robin_consumers():

            if remaining_tasks == 0:
                break

            if consumer_state[c] == IDLE:

                task = fetch_next_task()

                publish_task_descriptor(c, task)

                consumer_state[c] = PROCESSING
                consumer_next_chunk[c] = 0

                remaining_tasks--

        // 2. 为正在处理任务的consumer填充下一个chunk
        made_progress = false

        for c in round_robin_consumers():

            if consumer_state[c] != PROCESSING:
                continue

            if mailbox_not_consumed(c):
                continue

            chunk = consumer_next_chunk[c]

            if chunk >= consumer_total_chunks[c]:
                continue

            copy_chunk_to_mailbox(c, chunk)

            publish_mailbox_ready(c, chunk)

            consumer_next_chunk[c]++

            made_progress = true

        // 3. 回收已完成consumer
        for c in consumers:

            if consumer_state[c] == DONE:
                consumer_state[c] = IDLE
                made_progress = true

        if not made_progress:
            producer_backoff()

    publish_producer_done()
```

---

## 12.3 Consumer主循环

```cpp
function consumer_loop(c):

    observed_task_epoch = 0

    while true:

        wait_for_task_or_producer_done(c)

        if producer_done and no_pending_task(c):
            break

        descriptor = acquire_task_descriptor(c)

        hash_prepare(c)

        total_chunks =
            ceil_div(descriptor.edge_count, 64)

        for chunk in 0 .. total_chunks - 1:

            wait_mailbox_ready(c, chunk)

            edge_count =
                mailbox_chunk_size[c]

            for i = lane;
                i < edge_count;
                i += 32:

                post =
                    mailbox_post[c][i]

                weight =
                    mailbox_weight[c][i]

                if owner filtering enabled:
                    owner =
                        mailbox_owner[c][i]

                    if not spike_mask_contains(owner):
                        continue

                hash_insert_or_fallback(
                    c,
                    post,
                    weight
                )

            __syncwarp()

            release_mailbox(c, chunk)

        flush_hash(c, descriptor.batch)

        publish_consumer_done(c)
```

---

## 12.4 Mailbox copy

同步连续copy版本：

```cpp
function copy_chunk_to_mailbox(c, chunk):

    source_begin =
        task_offset[c] + chunk * 64

    remaining =
        task_edge_count[c] - chunk * 64

    count =
        min(64, remaining)

    for i = lane; i < count; i += 32:

        mailbox_post[c][i] =
            packed_post[source_begin + i]

        mailbox_weight[c][i] =
            packed_weight[source_begin + i]

        if owner enabled:
            mailbox_owner[c][i] =
                packed_owner[source_begin + i]

    __syncwarp()

    if lane == 0:
        mailbox_count[c] = count
```

---

# 13. Hash逻辑

现有最佳hash配置保持不变：

```text
aggregation = 512
capacity = 512
max probes = 4
min edges = 256
```

但 `min_edges` 判断基于完整 task：

```cpp
if task_edge_count < 256:
    use direct atomic path
```

对于 direct atomic task，也可通过 mailbox传递，但第一版更建议：

```text
小于256 edge的任务继续走原 consumer-direct路径
```

只有达到hash门槛的任务进入 producer-mailbox路径。

这样避免：

* 小任务被拆成 mailbox后固定成本过高；
* 低工作量无法覆盖同步开销；
* 128-edge hash负收益再次出现。

先前实验已经表明，128-edge聚合的理论重复率仅约1.09%，实际慢于baseline；512-edge才形成稳定收益。

---

# 14. Shared memory预算

## 14.1 Hash

七个consumer hash：

```text
keys:
7 × 512 × 4 B = 14 KB

values:
7 × 512 × 4 B = 14 KB

used slots:
7 × 512 × 2 B = 7 KB
```

合计：

```text
35 KB
```

## 14.2 Mailbox

```text
post:
7 × 64 × 4 B = 1.75 KB

weight:
7 × 64 × 4 B = 1.75 KB
```

合计：

```text
3.5 KB
```

加 metadata和barrier：

```text
约1 KB
```

预计总量：

```text
39.5–42 KB/block
```

目标：

```text
< 48 KB/block
```

应能维持2 blocks/SM。

如果超过限制，优先顺序：

1. used-slot list改bitmap；
2. mailbox size从64降到32；
3. 压缩owner metadata；
4. 减少冗余descriptor；
5. 最后才调整hash结构。

不能首先把512 aggregation降回256。

---

# 15. 避免轮询风暴

## 15.1 Producer等待

producer扫描七个consumer时，使用普通 acquire load，不执行CAS竞争。

若没有进展：

```cpp
__nanosleep(backoff);
```

backoff可以逐步增加：

```text
32
64
128 cycles
```

## 15.2 Consumer等待

consumer只等待两个变量：

```text
task_epoch[c]
mailbox_ready_epoch[c]
```

不会读取其他consumer状态。

## 15.3 统计等待周期，而不是每次循环atomicAdd

性能版本中不应每次等待都更新global统计。

建议：

```cpp
local_wait_counter++
```

任务结束或kernel结束时再写回一次。

否则统计本身可能改变结果。

---

# 16. 调度策略

第一版使用静态轮转：

```text
task 0 → consumer 0
task 1 → consumer 1
...
task 6 → consumer 6
task 7 → 第一个完成的consumer
```

不能固定：

```text
task 7必须等consumer 0
```

producer应扫描所有consumer，选择任意IDLE者。

这避免上一版的队头阻塞问题。

后续可加入：

```text
优先给刚释放mailbox的consumer继续发送当前task chunk
```

但第一版不做复杂优先队列。

---

# 17. 第一版版本拆分

必须依次建立以下版本。

## B0：当前最佳hash基线

```text
8 warp
每warp直接领取512-edge task
每warp自行load/hash/flush
```

## B1：7 consumer，无producer

```text
warp 0 idle
warp 1–7仍走原direct路径
```

目的：

* 单独量化少一个compute warp的损失；
* 确定7 consumer的理论下限。

如果 B1 已比 B0慢很多，需要把该损失计入后续目标。

## B2：7 consumer + 静态任务mailbox，不搬edge

producer只分发descriptor：

```text
consumer仍自己load edge
```

目的：

* 验证固定ownership；
* 验证任务状态机；
* 验证不竞争queue是否有收益或损失；
* 不引入edge staging。

## B3：64-edge同步连续mailbox

producer搬连续edge chunk，consumer hash。

目的：

* 验证短生命周期mailbox；
* 验证producer吞吐；
* 验证七consumer能同时工作。

## B4：64-edge异步bulk copy

目的：

* 测试异步copy本身收益。

## B5：TMA mailbox

目的：

* 测量TMA相对普通bulk async的增量。

每一步都必须与前一步比较，不能只比较B0和B5。

---

# 18. 正确性测试

## 18.1 Task级一致性

验证：

```text
B0 task count
=
producer assigned task count
=
consumer completed task count
```

验证：

```text
B0 input edge count
=
producer logical edge count
=
consumer processed active edge count
```

如果使用owner过滤，还需分别统计：

```text
loaded edges
active edges
filtered inactive edges
```

---

## 18.2 单任务测试

构造：

1. 1 edge；
2. 63 edges；
3. 64 edges；
4. 65 edges；
5. 511 edges；
6. 512 edges；
7. 多个重复post；
8. 无重复post；
9. hash probe失败；
10. 最后一个partial chunk。

---

## 18.3 并发测试

构造：

1. 七个consumer同时活跃；
2. consumer 0非常慢，其他consumer较快；
3. 一个consumer global flush很长；
4. producer无任务但consumer仍处理；
5. producer完成后最后一个mailbox仍未消费；
6. consumer处理完成后producer重新分配新任务；
7. 多timestep重复使用epoch。

---

## 18.4 数值比较

主要比较：

```text
B0 vs 新版本
```

而不是只比较dense reference。

逐项检查：

```text
spike mismatch
PSC max abs diff
PSC mean abs diff
PSC sum
nonzero PSC count
v max abs diff
```

建议先从：

```text
1 timestep
```

开始，再扩展到：

```text
8
32
128 timesteps
```

---

# 19. 运行时统计

记录：

```text
tasks assigned per consumer
tasks completed per consumer

chunks produced
chunks consumed

loaded edges
active edges
filtered edges

producer idle iterations
consumer task-wait iterations
consumer mailbox-wait iterations

consumer hash cycles
consumer flush cycles

mailbox occupancy
maximum simultaneous active consumers
```

最关键的新指标是：

```text
maximum simultaneous active consumers
```

目标：

```text
通常达到5–7
```

如果长期只有1–2，说明新设计仍然没有恢复并行度。

---

# 20. NCU检查指标

## 20.1 资源

```text
shared memory/block
registers/thread
active blocks/SM
active warps/SM
```

硬约束：

```text
2 blocks/SM
```

## 20.2 调度

```text
eligible warps/scheduler
issue active
active warps/scheduler
```

## 20.3 Stall

```text
Long Scoreboard
Barrier
MIO Throttle
LG Throttle
Wait
```

## 20.4 Shared路径

```text
shared load/store throughput
shared atomic throughput
bank conflicts
```

新版本不应出现大量stage-state atomic。

---

# 21. 成功标准

## B2任务分发版本

必须满足：

```text
性能不低于B1超过3%
```

否则说明状态机本身过重。

## B3同步mailbox版本

必须满足：

```text
最大同时active consumer >= 5
producer wait和consumer wait均无极端失衡
2 blocks/SM保持
```

性能至少应：

```text
不比B1慢超过5%
```

否则不进入TMA阶段。

## B4异步copy版本

目标：

```text
相对B3加速 >= 3%
```

或：

```text
Long Scoreboard明显下降
eligible warps明显提高
```

## B5 TMA版本

目标：

```text
相对B0最终加速 >= 5%
```

如果 TMA相对B4小于1–2%，说明bulk async已覆盖主要收益，没必要继续增加TMA复杂度。

---

# 22. 停止条件

出现以下情况时停止这一方向：

1. 连续packed layout本身显著增加总读取量；
2. active edge ratio长期低于50%；
3. B3恢复七consumer后仍明显慢于B1；
4. producer持续成为瓶颈；
5. mailbox size在32/64/128 sweep中均无正结果；
6. 异步copy无法降低consumer等待；
7. shared memory导致1 block/SM；
8. TMA相对普通async copy无增量；
9. pipeline仅把Long Scoreboard转化为Barrier或MIO stall。

---

# 23. 推荐首个落地版本

第一轮实际编码只实现：

```text
B1:
warp 0 idle
warp 1–7 original direct path

B2:
warp 0 only dispatches task descriptors
warp 1–7 still load/hash/flush themselves
per-consumer task mailbox
no edge staging
no TMA
```

这一步验证固定ownership与状态机。

第二轮再实现：

```text
B3:
64-edge continuous mailbox
7 fixed mailboxes
consumer retains hash across 8 chunks
mailbox released after each chunk
```

只有 B3 正确且性能接近 B1，才继续异步copy/TMA。

---

# 24. 最终架构摘要

```text
256-thread block

Warp 0:
    schedule tasks
    feed 7 fixed consumer mailboxes
    only perform continuous small-tile copy
    never execute map_packed_edge per edge

Warp 1–7:
    each owns one mailbox
    each owns one 512-entry hash
    each processes one 512-edge task
    receives 8 × 64-edge chunks
    releases mailbox after every chunk
    flushes hash after full task

Shared memory:
    7 private hash tables
    7 × 64-edge mailboxes
    per-consumer epochs and descriptors

Pipeline:
    producer fills mailbox for consumer B
    while consumer A hashes previous chunk
    while consumer C flushes completed task

Critical invariants:
    mailbox ownership is static
    no consumer competition
    no stage held during hash/flush
    producer does not perform sparse gather mapping
    5–7 consumers can remain active
    2 blocks/SM is preserved
```

新第一版的核心不是先“使用TMA”，而是先建立一个结构上正确的 warp specialization：

> Producer只提供低成本、连续、短生命周期的数据搬运；consumer仍保留完整的任务所有权、512-edge hash聚合和并行执行能力。

只有这个结构成立，TMA才能提供增量收益，而不会再次被任务串行化和stage占用问题淹没。
